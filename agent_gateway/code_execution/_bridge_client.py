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
from copy import deepcopy as _hank_deepcopy
from operator import index as _hank_index
from typing import Any as _HankAny, Protocol as _HankProtocol, SupportsIndex as _HankSupportsIndex

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


_HankPath = tuple[str | int | dict[str, list[int]], ...]
_HankLineage = tuple[str, tuple[str, ...], _HankPath]
_HankSlot = tuple[object, _HankLineage] | None


class _HankRefDict(dict):
  """A live producer object with an original operand reference per slot."""
  __slots__ = ("_hank_lineage", "_hank_slots", "_hank_original")
  _hank_lineage: _HankLineage | None
  _hank_slots: dict[object, _HankSlot]
  _hank_original: bool

  def __copy__(self) -> "_HankRefDict":
    carrier = type(self)()
    dict.update(carrier, self)
    carrier._hank_lineage = self._hank_lineage
    carrier._hank_original = self._hank_original
    carrier._hank_slots = self._hank_slots.copy()
    return carrier

  def __deepcopy__(self, memo: dict[int, object]) -> "_HankRefDict":
    carrier = type(self)()
    memo[id(self)] = carrier
    for key, child in self.items():
      copied_key = _hank_deepcopy(key, memo)
      copied_child = _hank_deepcopy(child, memo)
      dict.__setitem__(carrier, copied_key, copied_child)
      slot = self._hank_slots.get(key)
      carrier._hank_slots[copied_key] = (
        (copied_child, _hank_deepcopy(slot[1], memo))
        if slot is not None and child is slot[0] else None
      )
    carrier._hank_lineage = _hank_deepcopy(self._hank_lineage, memo)
    carrier._hank_original = self._hank_original
    return carrier

  def __reduce_ex__(self, protocol: _HankSupportsIndex, /):
    # Serialize origins, not a second operand tree: scalars need not retain
    # their identity through pickle, even when referenced more than once.
    origins: dict[object, _HankLineage | None] = {}
    for key, child in self.items():
      slot = self._hank_slots.get(key)
      origins[key] = slot[1] if slot is not None and child is slot[0] else None
    state = self._hank_lineage, self._hank_original, origins
    return type(self), (), state, None, iter(self.items())

  def __setstate__(
    self, state: tuple[_HankLineage | None, bool, dict[object, _HankLineage | None]],
  ) -> None:
    self._hank_lineage, self._hank_original, origins = state
    self._hank_slots = {
      key: None if origin is None else (self[key], origin)
      for key, origin in origins.items()
    }

  def __init__(self, *args, **kwargs):
    if not hasattr(self, "_hank_slots"):
      self._hank_lineage = None
      self._hank_slots = {}
      self._hank_original = False
    self.update(*args, **kwargs)

  def __setitem__(self, key, value):
    dict.__setitem__(self, key, value)
    self._hank_slots[key] = None
    self._hank_original = False

  def __delitem__(self, key):
    dict.__delitem__(self, key)
    self._hank_slots.pop(key, None)
    self._hank_original = False

  def update(self, other=(), /, **kwargs):
    if hasattr(other, "keys"):
      for key in other.keys():
        self[key] = other[key]
    else:
      for key, value in other:
        self[key] = value
    for key, value in kwargs.items():
      self[key] = value

  def setdefault(self, key, default=None, /):
    if key not in self:
      self[key] = default
    return self[key]

  def pop(self, key, /, *default):
    if key not in self:
      return dict.pop(self, key, *default)
    value = dict.pop(self, key, *default)
    self._hank_slots.pop(key, None)
    self._hank_original = False
    return value

  def popitem(self):
    key, value = dict.popitem(self)
    self._hank_slots.pop(key, None)
    self._hank_original = False
    return key, value

  def clear(self):
    dict.clear(self)
    self._hank_slots.clear()
    self._hank_original = False

  def __ior__(self, other):
    self.update(other)
    return self


class _HankRefList(list):
  """A live producer array; slot references move with their original operands."""
  __slots__ = ("_hank_lineage", "_hank_slots", "_hank_original")
  _hank_lineage: _HankLineage | None
  _hank_slots: list[_HankSlot]
  _hank_original: bool

  def __copy__(self) -> "_HankRefList":
    carrier = type(self)(self)
    carrier._hank_lineage = self._hank_lineage
    carrier._hank_original = self._hank_original
    carrier._hank_slots = self._hank_slots.copy()
    return carrier

  def __deepcopy__(self, memo: dict[int, object]) -> "_HankRefList":
    carrier = type(self)()
    memo[id(self)] = carrier
    list.extend(carrier, (_hank_deepcopy(child, memo) for child in self))
    carrier._hank_slots = [
      (copied_child, _hank_deepcopy(slot[1], memo))
      if slot is not None and child is slot[0] else None
      for child, copied_child, slot in zip(self, carrier, self._hank_slots)
    ]
    carrier._hank_lineage = _hank_deepcopy(self._hank_lineage, memo)
    carrier._hank_original = self._hank_original
    return carrier

  def __reduce_ex__(self, protocol: _HankSupportsIndex, /):
    origins = [
      slot[1] if slot is not None and child is slot[0] else None
      for child, slot in zip(self, self._hank_slots)
    ]
    state = self._hank_lineage, self._hank_original, origins
    return type(self), (), state, iter(self)

  def __setstate__(
    self, state: tuple[_HankLineage | None, bool, list[_HankLineage | None]],
  ) -> None:
    self._hank_lineage, self._hank_original, origins = state
    self._hank_slots = [
      None if origin is None else (child, origin)
      for child, origin in zip(self, origins)
    ]

  def __init__(self, values=()):
    list.__init__(self, values)
    self._hank_lineage = None
    self._hank_slots = [None] * len(self)
    self._hank_original = False

  def __getitem__(self, key):
    value = list.__getitem__(self, key)
    if not isinstance(key, slice):
      return value
    slots = self._hank_slots[key]
    for index, child in enumerate(value):
      if slots[index] is None:
        origin = _hank_origin(child)
        if origin is not None:
          slots[index] = (child, origin)
    lineage = _hank_slice_lineage(self, key, value, slots)
    carrier = _HankRefList(value)
    carrier._hank_lineage = lineage
    carrier._hank_original = lineage is not None
    carrier._hank_slots = slots
    return carrier

  def __setitem__(self, key, value):
    if isinstance(key, slice):
      slots = list(value._hank_slots) if isinstance(value, _HankRefList) else None
      value = list(value)
      list.__setitem__(self, key, value)
      self._hank_slots[key] = [None] * len(value) if slots is None else slots
    else:
      key = _hank_index(key)
      list.__setitem__(self, key, value)
      self._hank_slots[key] = None
    self._hank_original = False

  def __delitem__(self, key):
    key = key if isinstance(key, slice) else _hank_index(key)
    list.__delitem__(self, key)
    del self._hank_slots[key]
    self._hank_original = False

  def append(self, value):
    list.append(self, value)
    self._hank_slots.append(None)
    self._hank_original = False

  def extend(self, values):
    if isinstance(values, _HankRefList):
      slots = list(values._hank_slots)
      list.extend(self, values)
      self._hank_slots.extend(slots)
      self._hank_original = False
    else:
      for value in values:
        self.append(value)

  def insert(self, index, value):
    index = _hank_index(index)
    list.insert(self, index, value)
    self._hank_slots.insert(index, None)
    self._hank_original = False

  def pop(self, index: _HankSupportsIndex = -1, /):
    index = _hank_index(index)
    value = list.pop(self, index)
    self._hank_slots.pop(index)
    self._hank_original = False
    return value

  def remove(self, value):
    del self[self.index(value)]

  def clear(self):
    list.clear(self)
    self._hank_slots.clear()
    self._hank_original = False

  def reverse(self):
    list.reverse(self)
    self._hank_slots.reverse()
    self._hank_original = False

  def sort(self, *, key=None, reverse=False):
    operands = list(zip(self, self._hank_slots))
    # Native sort still empties this list during callbacks and owns mutation errors.
    list.__setitem__(self, slice(None), operands)
    self._hank_original = False
    try:
      list.sort(self, key=lambda operand: operand[0] if key is None else key(operand[0]), reverse=reverse)
    finally:
      operands = list.__getitem__(self, slice(None))
      list.__setitem__(self, slice(None), [operand[0] for operand in operands])
      self._hank_slots = [operand[1] for operand in operands]

  def __iadd__(self, values):
    self.extend(values)
    return self

  def __imul__(self, count):
    count = _hank_index(count)
    list.__imul__(self, count)
    self._hank_slots *= count
    self._hank_original = False
    return self


class _HankInput(dict):
  """One selected operand; publication checks that its live value is unchanged."""
  __slots__ = ("_hank_operand",)
  _hank_operand: object

  @classmethod
  def _hank_reconstruct(cls) -> "_HankInput":
    carrier = dict.__new__(cls)
    carrier._hank_operand = object()
    return carrier

  def __copy__(self) -> "_HankInput":
    carrier = self._hank_reconstruct()
    dict.update(carrier, self)
    carrier._hank_operand = self._hank_operand
    return carrier

  def __deepcopy__(self, memo: dict[int, object]) -> "_HankInput":
    carrier = self._hank_reconstruct()
    memo[id(self)] = carrier
    original = "value" in self and self["value"] is self._hank_operand
    for key, child in self.items():
      dict.__setitem__(carrier, _hank_deepcopy(key, memo), _hank_deepcopy(child, memo))
    if original:
      carrier._hank_operand = carrier["value"]
    return carrier

  def __reduce_ex__(self, protocol: _HankSupportsIndex, /):
    original = "value" in self and self["value"] is self._hank_operand
    return self._hank_reconstruct, (), original, None, iter(self.items())

  def __setstate__(self, original: bool) -> None:
    self._hank_operand = self["value"] if original else object()

  def __init__(self, **reference):
    dict.__init__(self, reference)
    self._hank_operand = reference["value"]


def _hank_origin(value: object) -> _HankLineage | None:
  """Only an intact original container can stand for its whole producer value."""
  if not isinstance(value, (_HankRefDict, _HankRefList)) or not value._hank_original:
    return None
  slots = value._hank_slots
  if len(value) != len(slots):
    return None
  pairs = (
    ((child, value._hank_slots.get(key)) for key, child in value.items())
    if isinstance(value, _HankRefDict) else zip(value, value._hank_slots)
  )
  for child, slot in pairs:
    if slot is None or child is not slot[0]:
      return None
    if isinstance(child, (_HankRefDict, _HankRefList)) and _hank_origin(child) is None:
      return None
  return value._hank_lineage


def _hank_slot_origin(value: object, slot: _HankSlot) -> _HankLineage | None:
  """Membership never sources a replacement; a moved carrier owns its origin."""
  if isinstance(value, (_HankRefDict, _HankRefList)):
    lineage = _hank_origin(value)
    if lineage is None:
      return None
    return slot[1] if slot is not None and value is slot[0] else lineage
  return slot[1] if slot is not None and value is slot[0] else None


def _hank_slice_lineage(
  container: _HankRefList, key: slice, values: list[object], slots: list[_HankSlot],
) -> _HankLineage | None:
  # Slice paths store normalized range bounds, not raw Python slice syntax.
  lineage = container._hank_lineage
  if container._hank_original and lineage is not None:
    nested_call_id, source_refs, path = lineage
    return nested_call_id, source_refs, (*path, {"slice": list(key.indices(len(container)))})
  origins = [_hank_slot_origin(value, slot) for value, slot in zip(values, slots)]
  if not origins or origins[0] is None:
    return None
  nested_call_id, source_refs, path = origins[0]
  if not path or not isinstance(path[-1], int):
    return None
  indices = []
  for origin in origins:
    if origin is None:
      return None
    call_id, refs, child_path = origin
    if call_id != nested_call_id or refs != source_refs or child_path[:-1] != path[:-1]:
      return None
    if not child_path or not isinstance(child_path[-1], int):
      return None
    indices.append(child_path[-1])
  step = indices[1] - indices[0] if len(indices) > 1 else 1
  if not step or any(index != indices[0] + position * step for position, index in enumerate(indices)):
    return None
  stop = indices[-1] + (1 if step > 0 else -1)
  return nested_call_id, source_refs, (*path[:-1], {"slice": [indices[0], stop, step]})


def _hank_reference(value, lineage):
  if lineage is None:
    return {"nested_call_id": None, "source_refs": [], "path": [], "value": _hank_plain(value)}
  nested_call_id, source_refs, path = lineage
  return {
    "nested_call_id": nested_call_id, "source_refs": list(source_refs),
    "path": list(path), "value": _hank_plain(value),
  }


def _hank_carry(value, nested_call_id, source_refs, path=()):
  """Materialize each container once, retaining references rather than snapshots."""
  if isinstance(value, dict):
    carrier = _HankRefDict()
    carrier._hank_lineage = (nested_call_id, source_refs, path)
    carrier._hank_original = True
    for key, child in value.items():
      child_path = (*path, key)
      child = _hank_carry(child, nested_call_id, source_refs, child_path)
      dict.__setitem__(carrier, key, child)
      origin: _HankLineage = (nested_call_id, source_refs, child_path)
      carrier._hank_slots[key] = (child, origin)
    return carrier
  if isinstance(value, list):
    carrier = _HankRefList()
    carrier._hank_lineage = (nested_call_id, source_refs, path)
    carrier._hank_original = True
    for index, child in enumerate(value):
      child_path = (*path, index)
      child = _hank_carry(child, nested_call_id, source_refs, child_path)
      list.append(carrier, child)
      origin = (nested_call_id, source_refs, child_path)
      carrier._hank_slots.append((child, origin))
    return carrier
  return value


def _hank_plain(value):
  """Expose a numpy/pandas scalar's native JSON value without copying a tree."""
  item = getattr(value, "item", None)
  if callable(item) and not isinstance(value, (dict, list, tuple, str, bytes)):
    try:
      return item()
    except (TypeError, ValueError):
      return value
  return value




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

  `call`/`search`/`describe` invoke host operations. `input` selects retained
  producer operands and `derive` publishes their expression and issued sources.
  Other public attributes forward to granted code-callable operations or
  delegation operations such as `agent`, `wait`, `completion`, and `judge`.
  Dict/list results behave as Python containers while keeping their original
  reference across cells. `_bind_cell` changes only the transport and token.
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
    if name.startswith("_") or name == "code_execute":
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

  @staticmethod
  def carry(result, nested_call_id, source_refs):
    """Attach the parent-issued reference to a bridge or native producer result."""
    return _hank_carry(result, nested_call_id, tuple(source_refs))

  def input(self, container, key):
    """Select the current value together with its own original operand reference."""
    value = container[key]
    if isinstance(container, _HankRefList):
      lineage = (
        _hank_origin(value) if isinstance(key, slice)
        else _hank_slot_origin(value, container._hank_slots[_hank_index(key)])
      )
    elif isinstance(container, _HankRefDict):
      lineage = _hank_slot_origin(value, container._hank_slots.get(key))
    else:
      lineage = _hank_origin(value)
    return _HankInput(**_hank_reference(value, lineage))

  def derive(self, name, value, *, expression, inputs):
    """Publish a derived row; literal operands publish an input_unsourced finding."""
    references = []
    for item in inputs:
      if isinstance(item, _HankInput):
        operand = item["value"]
        changed = operand is not item._hank_operand or (
          isinstance(operand, (_HankRefDict, _HankRefList)) and _hank_origin(operand) is None
        )
        references.append(_hank_reference(operand, None) if changed else dict(item))
      else:
        references.append(_hank_reference(item, _hank_origin(item)))
    return self._request({
      "type": "derive", "name": name, "value": _hank_plain(value),
      "expression": expression, "inputs": references,
    })

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
    result = reply.get("result")
    source_refs = reply.get("source_refs")
    return result if source_refs is None else self.carry(result, reply["nested_call_id"], source_refs)

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
