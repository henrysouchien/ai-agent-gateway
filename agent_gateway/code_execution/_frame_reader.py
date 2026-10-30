"""Host frame reader: nested tool calls from a running `code_execute` cell.

A cell's bridged calls reach session tools over the persistent kernel runner's
stdin/stdout NDJSON pipe (local process or Docker container stdio), not a
socket or loopback endpoint
(execution-boundary design §8; clients plan §5.3/§5.5).
`FrameReader` is session-owned and rebound by each request bundle to its current
dispatcher. Each cell captures that binding and owns its background channel;
session disposal awaits every remaining registration. Each `code_execute` call
opens one `CellChannel`, registered under a call-scoped token for the
outer `tool_call_id`: a v2 native token minted by the installed gateway
claim-signing authority for the placed bundle's `storage_id` (else the
principal's `risk_user_id`), or a random nonce when no authority is installed
(degraded, non-edge). Backend activation begins only when the cell executes;
completion revokes it before output collection. The same token authenticates
native calls through `lookup_active_cell`, sharing the channel's grant, ordinal,
ledger and source settlement. A non-executing token answers `bad_cell_token`.

Frames, one JSON object per line. Child -> host, every frame carries
`call_id` (child correlation id) and `cell_token`:

- `catalog{parent_call_id, request: "search", query}` -> list of `{name, description}`
- `catalog{parent_call_id, request: "describe", operation}` -> `{name, description, input_schema}`
- `derive{parent_call_id, name, value, expression, inputs}` -> the published derived row

A successful producer reply carries `nested_call_id` and `source_refs`, the
issued keys of the source rows recorded by that call. Derivations resolve those
keys against this session's retained source rows, including earlier cells.

Host -> child replies echo `type` and `call_id` and carry `ok` plus `result` or
`error{code, message, next_call?}`.

An operation is callable iff `operation ∈ owning tool_grant ∩ operations
registered as code-callable` (execution-boundary design §8): the grant is the
bound dispatcher's enforced catalog for the turn (`tool_definitions`); the
code-callable set is the registration the product binds beside it. There is
no second grant surface and no on-call activation: a tool is callable once
the session's `load_tools` has put it in the turn's active definitions. There
is no mutation-class filter here; the dispatcher's own policy and validation run on
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

import copy
import asyncio
import hmac
import hashlib
import json
import logging
import os
import secrets
import threading
import time
import weakref
from typing import AbstractSet, Any, Callable, Dict, List, Literal, Mapping, Optional
from agent_workflow_contracts.models import AdmittedTenant
from agent_workflow_contracts.next_call import NextCall, next_call as build_next_call
from pydantic import BaseModel, ConfigDict, StrictInt, StrictStr, ValidationError

from ..claim_signing_authority import configured_gateway_claim_signing_authority
from hank_model_transport.policy_imports import load_server_policy_helpers, resolve_server_policy_tool_class
from ..sdk_runner_helpers import server_for_tool
from ..runner_background_tasks import jobs_wait
from ..task_registry import TaskState
from ..tool_dispatch_declarations import canonical_dispatch_tool_name
from hank_model_transport.user_runtime_credentials import load_user_runtime_credentials
from agent_gateway.transport_runtime import GATEWAY_SEND_PROMPT_RUNTIME

from hank_model_transport.capability_execution import (
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
from ._frame_codec import CellBudget, ProtocolError, decode_frame, encode_frame
from ._tool_defs import CODE_EXECUTION_DELEGATION_TOOL_DEFS
log = logging.getLogger(__name__)

CELL_TOKEN_ENV = "AGENT_CODE_EXECUTE_CELL_TOKEN"

# Product-owned `(operation, result) -> source handles` for one nested result,
# and the citation registry's identity for one handle (`sources` dedup key).
SourceExtractor = Callable[[str, Any], List[Dict[str, Any]]]
SourceKey = Callable[[Dict[str, Any]], Any]
CatalogMetadata = Callable[[str], Mapping[str, object]]
# Product-owned decision for one exact granted operation name.
CodeCallable = Callable[[str], bool]

# Catalog previews are presentation only; source identities are never capped.
_DESCRIPTION_PREVIEW_CHARS = 300

class _SlicePath(BaseModel):
  model_config = ConfigDict(extra="forbid", strict=True)
  slice: list[StrictInt]


class _DerivedInput(BaseModel):
  model_config = ConfigDict(extra="forbid", strict=True)
  nested_call_id: StrictStr | None
  source_refs: list[StrictStr]
  path: list[StrictStr | StrictInt | _SlicePath]
  value: Any


class _DeriveFrame(BaseModel):
  """The one parse boundary for cell-authored derived rows."""
  model_config = ConfigDict(extra="forbid", strict=True)
  type: Literal["derive"]
  call_id: Any
  cell_token: StrictStr
  parent_call_id: StrictStr | None = None
  name: StrictStr
  value: Any
  expression: StrictStr
  inputs: list[_DerivedInput]


RUN_AGENT_OPERATION = "run_agent"


# The bridge operations this reader answers beyond `tool_call`/`catalog`. One
# registration (`CODE_EXECUTION_DELEGATION_TOOL_DEFS`): the cell's client generates
# `hank.<name>(...)` from the registered rows, and the `code_execute`
# prompt names them from there. An op lowers to the same authority the direct door
# uses; it is never in a grant or an MCP policy set.
def _delegation_operation_rows() -> list[Dict[str, Any]]:
  """The bridge's registered delegation-operation rows, derived from the delegation tool defs."""
  return [
    {
      "name": row["name"],
      "arguments": list(row["input_schema"]["properties"].keys()),
      "description": row["description"],
      "input_schema": row["input_schema"],
    }
    for row in CODE_EXECUTION_DELEGATION_TOOL_DEFS
  ]


# Every cell-to-host request a cell may send: the protocol frames and one
# frame per registered delegation operation. A backend's frame reader routes
# exactly these to the bridge, so a new operation needs no reader edit.
REQUEST_FRAME_TYPES: frozenset[str] = frozenset(
  {"catalog", "tool_call", "derive", *(str(row["name"]) for row in CODE_EXECUTION_DELEGATION_TOOL_DEFS)}
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
      raise ToolInputPreparationError({"code": "bridge_unavailable", "message": f"{operation!r} is not code-callable here; call {operation!r} directly as a tool, not from inside a cell"})
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
    with self._cell._lock:
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
    self._extract_sources: SourceExtractor | None = None
    self._source_key: SourceKey | None = None
    self._code_callable: CodeCallable | None = None
    self._catalog_metadata: CatalogMetadata | None = None
    self._job_runner: JobRunner | None = None
    self._grant_hint: Callable[[str], str | None] | None = None
    self._host_tools: frozenset[str] = CODE_EXECUTION_TOOL_NAMES
    self._cells: Dict[str, CellChannel] = {}
    # Settled per-cell summaries (`nested_calls`, `sources`, `derived`) keyed by
    # `outer_call_id` (and `task_id` for background `code` jobs). Lifetime owner:
    # `FrameReader.aclose()` / `cleanup_code_execution(session)`.
    self._cell_records: Dict[str, Dict[str, Any]] = {}
    # Source rows outlive cells alongside the sticky kernel's original carriers.
    self._issued_sources: Dict[str, Dict[str, Any]] = {}
    self._lineage_lock = threading.RLock()
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

  def bind_grant_hint(self, grant_hint: Callable[[str], str | None] | None) -> None:
    """Bind the product's tool-to-pack hint; the latest bind wins."""
    self._grant_hint = grant_hint

  def bind_dispatcher(
    self,
    dispatcher: Any,
    *,
    tool_definitions: Callable[[], List[Dict[str, Any]]],
    extract_sources: SourceExtractor | None = None,
    source_key: SourceKey | None = None,
    code_callable: CodeCallable | None = None,
    catalog_metadata: CatalogMetadata | None = None,
    host_tools: AbstractSet[str] | None = None,
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
    self._host_tools = (
      frozenset(host_tools) if host_tools is not None else CODE_EXECUTION_TOOL_NAMES
    )
  def active_cells(self) -> tuple[str, ...]:
    """Outer `tool_call_id`s of the cells currently registered."""
    return tuple(cell.outer_call_id for cell in self._cells.values())

  def cell_record(self, call_or_task_id: str) -> Dict[str, Any]:
    """Host-side settled ledger (`nested_calls`, `sources`) for one cell."""
    record = self._cell_records.get(call_or_task_id)
    return copy.deepcopy(record) if record is not None else {}

  def open_cell(
    self,
    outer_call_id: str,
    *,
    abort_event: asyncio.Event | None,
    active: bool,
    tool_ctx: ToolExecutionContext | None = None,
  ) -> "CellChannel":
    """Register one cell under a fresh token."""
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
    self._cell_records.clear()
    with self._lineage_lock:
      self._issued_sources.clear()

  def is_registered(self, cell: "CellChannel") -> bool:
    return self._cells.get(cell.token) is cell and cell._active and not self._closed

  def _registered(self, cell: "CellChannel", frame: Dict[str, Any]) -> bool:
    token = frame.get("cell_token")
    if not isinstance(token, str) or token != cell.token or not self.is_registered(cell):
      return False
    parent = frame.get("parent_call_id")
    if parent is None:
      return True
    return isinstance(parent, str) and hmac.compare_digest(parent, cell.outer_call_id)

  def _release(self, cell: "CellChannel") -> None:
    if self._cells.get(cell.token) is cell:
      del self._cells[cell.token]
    summary = cell.summary()
    self._cell_records[cell.outer_call_id] = summary
    if cell.job is not None:
      self._cell_records[cell.job[1]] = summary


def _mint_cell_token(principal: Mapping[str, Any]) -> str:
  """One call-scoped token: v2 native when a storage identity is known.

  The storage identity is the placed bundle's `storage_id`, else the
  principal's positive-int `risk_user_id`; the installed authority refuses
  anything that is not a canonical positive decimal. With no authority, or
  with neither a bundle nor a positive risk id (the direct, non-edge mode),
  the token is a plain nonce; the cell registry owns liveness either way.
  """
  authority = configured_gateway_claim_signing_authority()
  if authority is None:
    return secrets.token_urlsafe(32)
  bundle = load_user_runtime_credentials(os.environ)
  if bundle is not None:
    return authority.mint_native_cell_token(bundle.storage_id)
  risk_user_id = principal.get("risk_user_id")
  if isinstance(risk_user_id, int) and not isinstance(risk_user_id, bool) and risk_user_id > 0:
    return authority.mint_native_cell_token(str(risk_user_id))
  return secrets.token_urlsafe(32)


class CellChannel:
  """One cell's registration, frame handler and nested-call ledger."""

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
    self.token = _mint_cell_token(reader._principal)
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
    self._extract_sources = reader._extract_sources
    self._source_key = reader._source_key
    self._code_callable = reader._code_callable
    self._grant_hint = reader._grant_hint
    self._job_runner = reader._job_runner
    self._catalog_metadata = reader._catalog_metadata
    self._host_tools = reader._host_tools
    self._grant: Dict[str, Dict[str, Any]] | None = None
    self.budget = CellBudget()
    # Request frames are answered by concurrent tasks and native calls arrive
    # on producer threads; every ordinal/ledger/source mutation and read takes
    # this lock.
    self._lock = threading.RLock()
    self._ordinal = 0
    self.ledger: List[Dict[str, Any]] = []
    self._sources: Dict[Any, Dict[str, Any]] = {}
    self.derived: List[Dict[str, Any]] = []
    self._inflight: set[asyncio.Task[Any]] = set()

  # --- backend-facing (`_backends._base.CellBridge`) -------------------------

  def capability_env(self) -> Dict[str, str]:
    """Per-cell env the runner applies beside the control frame's `cell_token`.

    The token itself rides the cell control frame (`cell_token`), which the
    runner exports as `CELL_TOKEN_ENV`; only the signed risk-job claim, bound
    to this token and grant, is env.
    """
    env: Dict[str, str] = {}
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

  def activate(self) -> None:
    """Begin execution only after the backend owns its execution slot."""
    self._active = True
    self._active = self._reader.is_registered(self)

  def deactivate(self) -> None:
    """Revoke authentication at execution end, retaining the result ledger."""
    self._active = False

  def close(self) -> None:
    """End the registration; idempotent."""
    self.deactivate()
    self._reader._release(self)

  async def aclose(self) -> None:
    """Close this registration, cancel nested work, and await cancellation."""
    self.close()
    current = asyncio.current_task()
    tasks = set(self._inflight)
    if current is not None:
      tasks.discard(current)
    for task in tasks:
      task.cancel()
    if tasks:
      await asyncio.gather(*tasks, return_exceptions=True)

  def summary(self) -> Dict[str, Any]:
    """Publish nested calls, flat primary sources and derived rows independently."""
    summary: Dict[str, Any] = {}
    with self._lock:
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
        envelope = encode_frame({**reply, "result": None})
        result_bytes = len(wire) - len(envelope) + 4
        with self._lock:
          for entry in self.ledger:
            if entry["nested_call_id"] == nested_id:
              entry["result_bytes"] = result_bytes
              break
      return wire
    except ProtocolError as exc:
      # One over-limit reply is this call's correlated protocol_error; only an
      # aggregate breach poisons the budget (`CellBudget.charge`).
      nested_id = reply.get("nested_call_id")
      with self._lock:
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
      try:
        derived_frame = _DeriveFrame.model_validate(frame)
      except ValidationError as exc:
        return _error_reply("derive_result", call_id, "protocol_error", str(exc))
      return self._derive(derived_frame)
    return _error_reply(frame_type, call_id, "protocol_error", f"unknown frame type {frame_type!r}")

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
    """The turn's active grant, cached until an in-cell `load_tools` succeeds."""
    if self._grant is None:
      self._grant = self._loaded_granted()
    return self._grant

  def _route_row(self, name: str, definition: Dict[str, Any]) -> Dict[str, Any]:
    metadata = self._catalog_metadata(name) if self._catalog_metadata is not None else {}
    return {
      **metadata,
      "name": name,
      "description": str(definition.get("description") or ""),
      "input_schema": metadata.get("input_schema") or definition.get("input_schema") or {},
    }

  def _refusal(self, dispatcher: Any, operation: str) -> Optional[str]:
    """Error code, or None when `operation ∈ grant ∩ code-callable` (§8)."""
    if operation in self._host_tools:
      return "unknown_operation"
    if operation not in self._granted():
      if (
        dispatcher.route_origin_for_tool(operation) is None
        and (self._grant_hint is None or self._grant_hint(operation) is None)
      ):
        return "unknown_operation"
      return "not_granted"
    if self._code_callable is None or not self._code_callable(operation):
      return "not_code_callable"
    return None

  def _new_entry(self, operation: Any) -> Dict[str, Any]:
    with self._lock:
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
    if "next_call" in frame:
      try:
        parsed_next_call = NextCall.model_validate(frame["next_call"])
      except ValidationError as exc:
        first_err = exc.errors()[0] if exc.errors() else {}
        loc = ".".join(str(p) for p in first_err.get("loc", ()))
        msg = str(first_err.get("msg", str(exc)))
        first_msg = f"next_call.{loc}: {msg}" if loc else msg
        return _error_reply("tool_result", call_id, "type_error", first_msg)
      operation = parsed_next_call.tool_name
      tool_input = dict(parsed_next_call.arguments)
    delegation_ops = {row["name"]: row["arguments"] for row in _delegation_operation_rows()}
    if isinstance(operation, str) and operation in delegation_ops:
      arguments = delegation_ops[operation]
      raw_args = frame.get("args") or []
      raw_kwargs = tool_input or {}
      if not isinstance(raw_args, list) or not isinstance(raw_kwargs, dict):
        return _error_reply("tool_result", call_id, "protocol_error", "tool_call needs a string operation and an object input")
      if len(raw_args) > len(arguments):
        return _error_reply(
          "tool_result", call_id, "type_error", f"hank.{operation} takes at most {len(arguments)} positional arguments",
        )
      delegation_frame: Dict[str, Any] = {
        "type": operation,
        "call_id": call_id,
        "cell_token": frame.get("cell_token"),
        "parent_call_id": frame.get("parent_call_id"),
      }
      delegation_frame.update(zip(arguments, raw_args))
      for key, value in raw_kwargs.items():
        if key not in arguments:
          return _error_reply(
            "tool_result", call_id, "type_error", f"hank.{operation} got an unexpected argument {key!r}; accepts {arguments}",
          )
        delegation_frame[key] = value
      return await self._answer_frame(delegation_frame)
    if frame.get("args"):
      return _error_reply(
        "tool_result", call_id, "type_error", f"hank.{operation} takes keyword arguments only",
      )
    entry = self._new_entry(operation)
    nested_call_id = entry["nested_call_id"]

    def _refuse(code: str, message: str, next_call: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
      entry["error_code"] = code
      reply = _error_reply("tool_result", call_id, code, message, next_call)
      reply["nested_call_id"] = nested_call_id
      return reply

    if not isinstance(operation, str) or not isinstance(tool_input, dict):
      return _refuse("protocol_error", "tool_call needs a string operation and an object input")
    dispatcher = self._dispatcher
    if dispatcher is None:
      return _refuse("bridge_unavailable", "no dispatcher is bound to this session; call the operation directly as a tool outside the cell, or run in a session with a tool dispatcher bound")
    refusal = self._refusal(dispatcher, operation)
    entry["server"] = _server_of(operation)
    entry["policy_class"] = _policy_class(operation)
    if refusal == "unknown_operation":
      return _refuse(refusal, f"{operation!r} is not a tool on this session's live surface")
    if refusal == "not_granted":
      pack = self._grant_hint(operation) if self._grant_hint is not None else None
      return _refuse(
        refusal,
        (
          f'{operation!r} is not granted to this turn; load its pack first: '
          f'hank.call("load_tools", pack="{pack}") (or call it directly as a tool)'
          if pack is not None else
          f"{operation!r} is not granted to this turn; load its pack with load_tools (or call it directly as a tool)"
        ),
        next_call=build_next_call("load_tools", {"pack": pack}) if pack is not None else None,
      )
    if refusal is not None:
      return _refuse(refusal, f"{operation!r} is not registered as code-callable")
    reply = await self._dispatch_granted("tool_result", call_id, entry, dispatcher, operation, tool_input)
    if operation == "load_tools" and entry["ok"]:
      self._grant = None
    return reply

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
      return _refuse("bridge_unavailable", "no dispatcher is bound to this session; call the agent tool directly outside the cell, or run in a session with delegation bound")
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
      return _error_reply("wait_result", call_id, "bridge_unavailable", "no job registry is bound to this session; call the outer jobs/wait tool directly outside the cell, or run in a session with delegation bound")
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
      return _error_reply("completion_result", call_id, "bridge_unavailable", "no job runner is bound to this session; call the outer completion tool directly outside the cell, or run in a session with delegation bound")

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
        runtime=GATEWAY_SEND_PROMPT_RUNTIME,
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
    return self._producer_reply("completion_result", call_id, launched, entry)

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
      return _error_reply("judge_result", call_id, "bridge_unavailable", "no job runner is bound to this session; call the outer judge tool directly outside the cell, or run in a session with delegation bound")

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
        runtime=GATEWAY_SEND_PROMPT_RUNTIME,
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
    return self._producer_reply("judge_result", call_id, launched, entry)

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
      return _refuse("bridge_unavailable", "no dispatcher is bound to this session; call the agent tool directly outside the cell, or run in a session with delegation bound")
    if RUN_AGENT_OPERATION not in self._granted():
      return _refuse("not_granted", f"{RUN_AGENT_OPERATION!r} is not granted to this turn")

    runner = self._job_runner() if self._job_runner is not None else None
    registry = getattr(runner, "_task_registry", None)
    if registry is None:
      return _refuse("bridge_unavailable", "no job registry is bound to this session; call the agent tool directly outside the cell, or run in a session with delegation bound")

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
    return self._producer_reply("workpool_result", call_id, result_payload, entry)

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
      return _refuse("bridge_unavailable", "no job registry is bound to this session; call the outer tool directly outside the cell, or run in a session with delegation bound")

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
          runtime=GATEWAY_SEND_PROMPT_RUNTIME,
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
      return self._producer_reply("judge_batch_result", call_id, {
        "operation": op_name,
        "status": batch_status,
        "judgments": judgments,
      }, entry)
    else:
      entry["duration_ms"] = _elapsed_ms(started)
      entry["ok"] = True
      entry["error_code"] = None
      return self._producer_reply("judge_batch_result", call_id, {
        "operation": op_name,
        "status": "running",
        "jobs": {k: {"task_id": tid, "status": "running"} for k, tid in task_id_by_key.items()},
      }, entry)
  def _producer_reply(
    self, reply_type: str, call_id: Any, result: Any, entry: Dict[str, Any],
  ) -> Dict[str, Any]:
    """Every producer returns its parent-issued reference, even without sources."""
    return {
      **_ok_reply(reply_type, call_id, result),
      "nested_call_id": entry["nested_call_id"],
      "source_refs": entry.setdefault("source_refs", []),
    }

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
    if settlement.outcome == "ok":
      entry["source_refs"] = self._record_sources(operation, nested_call_id, result, settlement.sources)
    return self._producer_reply(reply_type, call_id, result, entry)

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
      ref = _issued_ref(key)
      with self._lock:
        row = self._sources.setdefault(
          key, {**handle, "produced_by_tool": canonical_dispatch_tool_name(operation)},
        )
      with self._reader._lineage_lock:
        self._reader._issued_sources.setdefault(ref, row)
      if ref not in refs:
        refs.append(ref)
    return refs

  def _derive(self, frame: _DeriveFrame) -> Dict[str, Any]:
    """Publish selected observations; missing issuance is a finding, not refusal."""
    resolved: List[Dict[str, Any]] = []
    findings: List[Dict[str, Any]] = []
    with self._reader._lineage_lock:
      issued = self._reader._issued_sources
      for index, reference in enumerate(frame.inputs):
        refs = list(dict.fromkeys(reference.source_refs))
        sources = [issued[ref] for ref in refs if ref in issued]
        path = [part.model_dump() if isinstance(part, _SlicePath) else part for part in reference.path]
        row = {
          "nested_call_id": reference.nested_call_id, "path": path,
          "value": reference.value, "sources": sources,
        }
        resolved.append(row)
        reason = None
        if not refs:
          reason = "the input carries no source reference; select it from a producer result with hank.input"
        elif len(sources) != len(refs):
          reason = "the input's source references were not issued by this session"
        if reason is not None:
          findings.append({
            "code": "input_unsourced", "input": index,
            "nested_call_id": reference.nested_call_id, "path": path, "message": reason,
          })
    published = {
      "name": frame.name, "value": frame.value, "expression": frame.expression,
      "status": "derived", "inputs": resolved, "findings": findings,
    }
    with self._lock:
      self.derived.append(published)
    return _ok_reply("derive_result", frame.call_id, published)

  def _catalog(self, frame: Dict[str, Any]) -> Dict[str, Any]:
    call_id = frame.get("call_id")
    dispatcher = self._dispatcher
    if dispatcher is None or self._tool_definitions is None:
      return _error_reply("catalog", call_id, "bridge_unavailable", "no dispatcher is bound to this session; list tools via the outer tool catalog outside the cell, or run in a session with a tool dispatcher bound")
    request = frame.get("request")
    if request == "search":
      query = str(frame.get("query") or "").lower()
      rows = []
      for name, definition in self._granted().items():
        description = str(definition.get("description") or "")
        if query not in name.lower() and query not in description.lower():
          continue
        if self._code_callable is not None and self._code_callable(name):
          raw_metadata = self._catalog_metadata(name) if self._catalog_metadata is not None else {}
          metadata = {k: v for k, v in raw_metadata.items() if k != "input_schema"}
          rows.append({**metadata, "name": name, "description": description[:_DESCRIPTION_PREVIEW_CHARS]})
      return _ok_reply("catalog", call_id, rows)
    if request == "describe":
      operation = frame.get("operation")
      if not isinstance(operation, str):
        return _error_reply("catalog", call_id, "protocol_error", "describe needs a string operation")
      # A registered bridge op describes itself; it is called as
      # `hank.<name>(...)`, never through `hank.call`, so `search` (the
      # grant ∩ code-callable listing) does not carry it.
      for row in _delegation_operation_rows():
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


def _error_reply(
  frame_type: Any, call_id: Any, code: str, message: str, next_call: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
  error: Dict[str, Any] = {"code": code, "message": message}
  if next_call is not None:
    error["next_call"] = NextCall.model_validate(next_call).to_dict()
  return {"type": frame_type, "call_id": call_id, "ok": False, "error": error}
