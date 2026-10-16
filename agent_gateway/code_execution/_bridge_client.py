"""One stdlib-only hank client for inherited pipes and kernel transports.

Shipped as source in the cell preamble. The runner supplies a per-cell transport
and immutable token; the one-shot subprocess opens its inherited fd pair lazily.
Both paths use the same codec, handshake, correlation and error handling.
"""
import os as _hank_os
import secrets as _hank_secrets
import threading as _hank_threading
import importlib as _hank_importlib
from typing import Any as _HankAny, Protocol as _HankProtocol

try:
  _hank_codec = _hank_importlib.import_module("_hank_frame_codec")
except ImportError:
  _hank_codec = _hank_importlib.import_module("agent_gateway.code_execution._frame_codec")

class _HankTransport(_HankProtocol):
  def exchange(self, payload: dict[str, _HankAny], /) -> dict[str, _HankAny]: ...



class HankBridgeError(Exception):
  """A bridge request the host answered with an error frame."""

  def __init__(self, code, message=""):
    super().__init__(f"{code}: {message}" if message else str(code))
    self.code = code
    self.message = message


class _HankRefDict(dict):
  """A producer result's object that keeps `(nested_call_id, source_refs, path)`.

  Behaves as a plain dict for the cell's code; `hank.input` reads the reference.
  """
  __slots__ = ("_hank_lineage",)


class _HankRefList(list):
  """A producer result's array that keeps `(nested_call_id, source_refs, path)`.

  A slice is a carrier too; its path ends in `{"slice": [start, stop, step]}`.
  """
  __slots__ = ("_hank_lineage",)

  def __getitem__(self, key):
    value = list.__getitem__(self, key)
    lineage = getattr(self, "_hank_lineage", None)
    if not isinstance(key, slice) or lineage is None:
      return value
    nested_call_id, source_refs, path = lineage
    carrier = _HankRefList(value)
    carrier._hank_lineage = (nested_call_id, source_refs, (*path, {"slice": list(key.indices(len(self)))}))
    return carrier


class _HankInput(dict):
  """`{nested_call_id, source_refs, path, value}`: one selected producer value."""


def _hank_carry(value, nested_call_id, source_refs, path=()):
  """Rebuild a producer result with every object and array carrying its lineage."""
  if isinstance(value, dict):
    carrier = _HankRefDict()
    carrier._hank_lineage = (nested_call_id, source_refs, path)
    for key, child in value.items():
      dict.__setitem__(carrier, key, _hank_carry(child, nested_call_id, source_refs, path + (key,)))
    return carrier
  if isinstance(value, list):
    carrier = _HankRefList()
    carrier._hank_lineage = (nested_call_id, source_refs, path)
    for index, child in enumerate(value):
      list.append(carrier, _hank_carry(child, nested_call_id, source_refs, path + (index,)))
    return carrier
  return value


def _hank_plain(value):
  """A numpy/pandas scalar as the Python scalar the wire can carry."""
  item = getattr(value, "item", None)
  if callable(item) and not isinstance(value, (dict, list, tuple, str, bytes)):
    try:
      return item()
    except (TypeError, ValueError):
      return value
  return value


class _PipeTransport:
  def __init__(self, reader, writer):
    self.reader = reader
    self.writer = writer
    self.budget = _hank_codec.CellBudget()

  def exchange(self, payload):
    self.writer.write(_hank_codec.encode_frame(payload, self.budget))
    self.writer.flush()
    line = self.reader.readline(_hank_codec.MAX_FRAME_BYTES + 1)
    if not line:
      raise HankBridgeError("bridge_closed", "the host closed the bridge channel")
    return _hank_codec.decode_frame(line, self.budget)


class HankBridge:
  """Calls the current cell's granted, code-callable operations.

  `call`/`search`/`describe` are fixed. The host's `hello` reply registers the
  other bridge operations (`agent`, `wait`, `completion`, `judge`, ...); each
  becomes a method that frames its keyword arguments under the registered name,
  so the cell never carries a hand-written copy of that catalog.
  """

  def __init__(self, transport: _HankTransport | None = None, token=None):
    self._transport = transport
    self._token = token
    self._parent_call_id = None
    self._operations = {}
    self._seq = 0
    self._nonce = _hank_secrets.token_hex(12)
    self._opened = False
    self._lock = _hank_threading.RLock()

  def __getattr__(self, name):
    if name.startswith("_"):
      raise AttributeError(name)
    with self._lock:
      self._open()
      registered = self._operations.get(name)
    if registered is None:
      raise AttributeError(f"hank has no operation {name!r}; registered: {sorted(self._operations)}")
    arguments = list(registered.get("arguments") or [])

    def operation(*args, **kwargs):
      if len(args) > len(arguments):
        raise TypeError(f"hank.{name} takes at most {len(arguments)} positional arguments")
      frame = {"type": name}
      frame.update(zip(arguments, args))
      for key, value in kwargs.items():
        if key not in arguments:
          raise TypeError(f"hank.{name} got an unexpected argument {key!r}; accepts {arguments}")
        frame[key] = value
      return self._request(frame)

    operation.__name__ = name
    operation.__doc__ = registered.get("description")
    return operation

  def call(self, operation, **arguments):
    """Run one code-callable operation; returns its result payload."""
    return self._request({"type": "tool_call", "operation": operation, "input": arguments})

  @staticmethod
  def carry(result, nested_call_id, source_refs):
    """Return a producer `result` as a lineage carrier for `hank.input`/`hank.derive`.

    `nested_call_id` and `source_refs` are the parent-issued reference of the call
    that produced it. Bridge replies and native domain clients both come through here.
    """
    return _hank_carry(result, nested_call_id, tuple(source_refs))

  def input(self, container, key):
    """Select `container[key]` as an input for `hank.derive`.

    `container` is a producer result, or a nested object/array kept from one
    (any earlier cell's too). The returned `{nested_call_id, source_refs, path,
    value}` takes its `value` from the retained result. A container that did not
    come from a producer carries no reference.
    """
    value = container[key]
    lineage = getattr(container, "_hank_lineage", None)
    if lineage is None:
      return _HankInput(nested_call_id=None, source_refs=[], path=[], value=_hank_plain(value))
    nested_call_id, source_refs, path = lineage
    if isinstance(key, slice):
      step = list(key.indices(len(container)))
      element = {"slice": step}
    elif isinstance(key, int) and not isinstance(key, bool) and isinstance(container, list):
      element = key + len(container) if key < 0 else key
    else:
      element = key
    return _HankInput(
      nested_call_id=nested_call_id, source_refs=list(source_refs), path=[*path, element], value=value,
    )

  def derive(self, name, value, *, expression, inputs):
    """Publish `value` as derived from `inputs` by `expression`.

    `inputs` are `hank.input(...)` references; a producer object/array given
    directly stands for itself, and anything else is a literal with no source.
    The row, with each input's issued source rows, joins this cell's result as
    `derived`; it is never a source itself.
    """
    references = []
    for item in inputs:
      if isinstance(item, _HankInput):
        references.append(dict(item))
      else:
        lineage = getattr(item, "_hank_lineage", None)
        if lineage is None:
          references.append({"nested_call_id": None, "source_refs": [], "path": [], "value": _hank_plain(item)})
        else:
          nested_call_id, source_refs, path = lineage
          references.append(
            {"nested_call_id": nested_call_id, "source_refs": list(source_refs), "path": list(path), "value": item},
          )
    return self._request({
      "type": "derive", "name": name, "value": _hank_plain(value), "expression": expression, "inputs": references,
    })

  def search(self, query):
    """Code-callable operations whose name or description mentions query."""
    return self._request({"type": "catalog", "request": "search", "query": query})

  def describe(self, operation):
    """Name, description and input schema of one code-callable operation."""
    return self._request({"type": "catalog", "request": "describe", "operation": operation})

  def _request(self, frame):
    with self._lock:
      transport = self._open()
      reply = self._exchange(transport, {**frame, "parent_call_id": self._parent_call_id})
    result = reply.get("result")
    source_refs = reply.get("source_refs")
    if source_refs is None:
      return result
    return self.carry(result, reply.get("nested_call_id"), source_refs)

  def _open(self) -> _HankTransport:
    if self._transport is None:
      fd_in = _hank_os.environ.get("AGENT_CODE_EXECUTE_BRIDGE_FD_IN")
      fd_out = _hank_os.environ.get("AGENT_CODE_EXECUTE_BRIDGE_FD_OUT")
      self._token = _hank_os.environ.get("AGENT_CODE_EXECUTE_CELL_TOKEN")
      if not (fd_in and fd_out and self._token):
        raise HankBridgeError("bridge_unavailable", "hank is only available inside a bridged code_execute cell")
      try:
        self._transport = _PipeTransport(_hank_os.fdopen(int(fd_in), "rb"), _hank_os.fdopen(int(fd_out), "wb"))
      except (OSError, ValueError) as exc:
        raise HankBridgeError("bridge_unavailable", str(exc)) from None
    transport = self._transport
    if not self._opened:
      hello = self._exchange(transport, {"type": "hello", "protocol": 1}).get("result")
      if isinstance(hello, dict):
        self._parent_call_id = hello.get("parent_call_id")
        self._operations = {
          str(row["name"]): row
          for row in (hello.get("operations") or [])
          if isinstance(row, dict) and row.get("name")
        }
      self._opened = True
    return transport

  def _exchange(self, transport: _HankTransport, frame):
    self._seq += 1
    call_id = f"{self._nonce}.{self._seq}"
    payload = {**frame, "call_id": call_id, "cell_token": self._token}
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
    if not reply.get("ok"):
      raise HankBridgeError(str(error.get("code") or "bridge_error"), str(error.get("message") or ""))
    return reply


hank = HankBridge(transport=globals().get("_hank_transport"), token=globals().get("_hank_cell_token"))
