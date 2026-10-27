"""One stdlib-only hank client over an injected transport.

Shipped as source in the cell preamble. The kernel keeps one session-sticky
`HankBridge` and binds each cell's transport and immutable token; every call
goes through the same codec, correlation and error handling.
"""
import asyncio as _hank_asyncio
import contextvars as _hank_contextvars
import os as _hank_os
import secrets as _hank_secrets
import threading as _hank_threading
import importlib as _hank_importlib
from collections.abc import Mapping as _HankMapping
from typing import Any as _HankAny, Protocol as _HankProtocol

try:
  _hank_codec = _hank_importlib.import_module("_hank_frame_codec")
except ImportError:
  _hank_codec = _hank_importlib.import_module("agent_gateway.code_execution._frame_codec")

# True while an `acall` waits on the host: the transport pauses the cell clock.
_HANK_PAUSES_CELL = _hank_contextvars.ContextVar("_HANK_PAUSES_CELL", default=False)


class _HankTransport(_HankProtocol):
  def exchange(self, payload: dict[str, _HankAny], /) -> dict[str, _HankAny]: ...



class HankBridgeError(Exception):
  """A host error frame, with optional recovery usable via hank.call(exc.next_call)."""

  def __init__(self, code, message="", next_call=None):
    super().__init__(f"{code}: {message}" if message else str(code))
    self.code = code
    self.message = message
    self.next_call = next_call



def _hank_call_target(operation, args, arguments):
  """Resolve `call`'s three spellings to (operation name, args, input arguments)."""
  if isinstance(operation, str):
    if not operation:
      raise TypeError("hank.call needs a non-empty operation name")
    return operation, args, arguments
  if operation is None and not args:
    next_call = arguments
  elif isinstance(operation, _HankMapping) and not args and not arguments:
    next_call = operation
  else:
    raise TypeError("hank.call takes an operation name, or a next_call mapping alone")
  if "tool_name" not in next_call:
    raise TypeError("hank.call next_call needs 'tool_name' naming the operation")
  name = next_call["tool_name"]
  if not isinstance(name, str) or not name:
    raise TypeError("hank.call next_call 'tool_name' must be a non-empty operation name")
  inner = next_call.get("arguments")
  if inner is None:
    inner = {}
  if not isinstance(inner, _HankMapping) or not all(isinstance(key, str) for key in inner):
    raise TypeError("hank.call next_call 'arguments' must be an object")
  return name, (), dict(inner)


class HankBridge:
  """Calls the current cell's granted, code-callable operations.

  `call`/`search`/`describe` are fixed. Every other public attribute is
  an operation forwarder: `hank.<name>(*args, **kwargs)` frames the call under
  `name` and the host resolves it (a delegation operation such as `agent`,
  `wait`, `completion`, `judge`, or a granted code-callable tool). Results are
  plain Python values; provenance is recorded host-side on the cell's
  nested-call ledger. One instance serves every cell of a kernel process;
  `_bind_cell` points it at the current cell's transport and token.
  """

  def __init__(self, transport: _HankTransport | None = None, token=None):
    self._transport = transport
    self._token = token
    self._parent_call_id = None
    self._seq = 0
    self._nonce = _hank_secrets.token_hex(12)
    self._lock = _hank_threading.RLock()

  def _bind_cell(self, transport: _HankTransport | None = None, token=None, parent_call_id=None):
    with self._lock:
      self._transport = transport
      self._token = token
      self._parent_call_id = parent_call_id
    return self

  def __getattr__(self, name):
    if name.startswith("_") or name in ("code_execute", "carry", "input", "derive"):
      raise AttributeError(name)

    def operation(*args, **kwargs):
      return self.call(name, *args, **kwargs)

    operation.__name__ = name
    return operation

  def call(self, operation=None, /, *args, **arguments):
    """Run one code-callable operation; returns its plain result payload.

    `hank.call("name", *args, **kwargs)`; or a `next_call` mapping
    (`{"tool_name": name, "arguments": {...}}`), passed positionally or
    unpacked as `hank.call(**next_call)`. A positional or keyword `operation=`
    after the name is an ordinary argument of the called operation.
    """
    raw_next_call = (
      arguments
      if operation is None and not args
      else operation
      if isinstance(operation, _HankMapping) and not args and not arguments
      else None
    )
    operation, args, arguments = _hank_call_target(operation, args, arguments)
    frame: dict[str, _HankAny] = {"type": "tool_call", "operation": operation, "input": arguments}
    if raw_next_call is not None:
      frame["next_call"] = dict(raw_next_call)
    if args:
      frame["args"] = list(args)
    return self._request(frame)

  async def acall(self, operation=None, /, *args, **arguments):
    """`call` from async code: the blocking exchange runs on a worker thread.

    The cell's clock is paused for the wait (`_HANK_PAUSES_CELL` is True in the
    worker's context, which `asyncio.to_thread` copies from this task).
    """
    token = _HANK_PAUSES_CELL.set(True)
    try:
      return await _hank_asyncio.to_thread(self.call, operation, *args, **arguments)
    finally:
      _HANK_PAUSES_CELL.reset(token)

  def search(self, query, /):
    """Code-callable operations whose name or description mentions query."""
    return self._request({"type": "catalog", "request": "search", "query": query})

  def describe(self, operation, /):
    """Name, description and input schema of one code-callable operation."""
    return self._request({"type": "catalog", "request": "describe", "operation": operation})

  def _request(self, frame):
    # The lock covers only the binding snapshot and the sequence number; the
    # exchange itself runs unlocked so concurrent calls overlap on the wire.
    with self._lock:
      transport = self._transport
      token = self._token
      parent_call_id = self._parent_call_id or _hank_os.environ.get("AGENT_TELEMETRY_TOOL_CALL_ID")
      self._seq += 1
      call_id = f"{self._nonce}.{self._seq}"
    if transport is None:
      raise HankBridgeError("bridge_unavailable", "hank is only available inside a bridged code_execute cell")
    reply = self._exchange(transport, {**frame, "parent_call_id": parent_call_id, "call_id": call_id, "cell_token": token})
    return reply.get("result")

  def _exchange(self, transport: _HankTransport, payload):
    call_id = payload["call_id"]
    try:
      reply = transport.exchange(payload)
    except _hank_codec.ProtocolError as exc:
      raise HankBridgeError("protocol_error", str(exc)) from None
    except OSError as exc:
      raise HankBridgeError("bridge_closed", str(exc)) from None
    if not isinstance(reply, dict):
      raise HankBridgeError("protocol_error", "reply is not an object")
    error = reply.get("error") or {}
    if not isinstance(error, dict):
      raise HankBridgeError("protocol_error", "reply error is not an object")
    if reply.get("call_id") != call_id and not (
      reply.get("call_id") is None and error.get("code") == "protocol_error"
    ):
      raise HankBridgeError("protocol_error", f"reply for {reply.get('call_id')!r}, expected {call_id!r}")
    next_call = error.get("next_call")
    if next_call is not None:
      if not isinstance(next_call, _HankMapping):
        raise HankBridgeError("protocol_error", "malformed next_call: expected a mapping or None")
      try:
        _hank_call_target(next_call, (), {})
      except TypeError as exc:
        raise HankBridgeError("protocol_error", f"malformed next_call: {exc}") from None
    if not reply.get("ok"):
      if error.get("code") == "type_error":
        raise TypeError(str(error.get("message") or ""))
      raise HankBridgeError(
        str(error.get("code") or "bridge_error"),
        str(error.get("message") or ""),
        next_call,
      )
    return reply
