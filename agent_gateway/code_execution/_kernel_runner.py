"""Dependency-free persistent interpreter, shipped with the shared codec/client.

Control cells and bridge replies arrive on stdin. Stdout is exclusively bounded
NDJSON; Python, native and descendant output use separate capture pipes. The
same source runs locally and as Docker's long-lived interpreter. SIGINT ends a
cell, not the namespace; EOF/session disposal ends the interpreter.
"""
import base64
import codecs
import importlib
import io
import os
import queue
import select
import signal
import sys
import threading
import time
import traceback
from typing import TypedDict

# Package imports use the source owners; the standalone launcher installs the
# same modules before entering this runner, without a gateway dependency.
if __package__:
  from . import _bridge_client, _frame_codec as _codec
else:
  _bridge_client = importlib.import_module("_hank_bridge_client")
  _codec = importlib.import_module("_hank_frame_codec")
CellBudget = _codec.CellBudget
MAX_CELL_BYTES = _codec.MAX_CELL_BYTES
MAX_FRAME_BYTES = _codec.MAX_FRAME_BYTES
ProtocolError = _codec.ProtocolError
decode_frame = _codec.decode_frame
encode_frame = _codec.encode_frame


class _ActiveCell(TypedDict):
  id: str | None
  budget: CellBudget | None
  fault: BaseException | None


class _FrameBuffer:
  """One capture stream's bounded line buffer."""

  def __init__(self, emit, cell_id, name, chunk_size):
    self._emit = emit
    self._cell_id = cell_id
    self._name = name
    self._chunk_size = chunk_size
    self._text = ""
    self._lock = threading.RLock()


  def write(self, text):
    if not isinstance(text, str):
      raise TypeError("write() argument must be str")
    with self._lock:
      for offset in range(0, len(text), self._chunk_size):
        self._text += text[offset:offset + self._chunk_size]
        while self._text:
          newline = self._text.find("\n")
          length = min(newline + 1 if newline >= 0 else self._chunk_size, self._chunk_size)
          if newline < 0 and len(self._text) < length:
            break
          part, self._text = self._text[:length], self._text[length:]
          self._emit({"type": self._name, "id": self._cell_id, "text": part})
    return len(text)

  def flush(self):
    with self._lock:
      if self._text:
        text, self._text = self._text, ""
        self._emit({"type": self._name, "id": self._cell_id, "text": text})


class _Stream(io.TextIOBase):
  """Python and native output share the same pipes and ordering owner."""

  def __init__(self, cell_id, fd, origin, flush_capture):
    self._cell_id = cell_id
    self._fd = fd
    self._origin = origin
    self._flush_capture = flush_capture

  encoding: str = "utf-8"

  def writable(self):
    return True

  def fileno(self):
    return self._fd

  def write(self, text):
    if not isinstance(text, str):
      raise TypeError("write() argument must be str")
    if self._origin() != self._cell_id:
      return len(text)
    for offset in range(0, len(text), 8192):
      remaining = memoryview(text[offset:offset + 8192].encode("utf-8"))
      while remaining:
        remaining = remaining[os.write(self._fd, remaining):]
    return len(text)

  def flush(self):
    if self._origin() == self._cell_id:
      self._flush_capture()


class _NoInput(io.TextIOBase):
  def read(self, *args):
    raise RuntimeError("interactive stdin is not supported")

  readline = read
  readlines = read


def _mime_bundle(value):
  bundle: dict[str, object] = {"text/plain": repr(value)}
  if isinstance(value, (dict, list, tuple)):
    try:
      import json
      bundle["application/json"] = json.loads(json.dumps(value))
    except (TypeError, ValueError, OverflowError):
      pass
  for method_name, mime in (
    ("_repr_png_", "image/png"),
    ("_repr_html_", "text/html"),
    ("_repr_markdown_", "text/markdown"),
    ("_repr_json_", "application/json"),
  ):
    method = getattr(value, method_name, None)
    if callable(method):
      representation = method()
      if representation is not None:
        if mime == "image/png" and isinstance(representation, (bytes, bytearray)):
          representation = base64.b64encode(representation).decode("ascii")
        bundle[mime] = representation
  if any(cls.__module__.startswith("matplotlib.") for cls in type(value).__mro__):
    try:
      figure_type = importlib.import_module("matplotlib.figure").Figure
    except ImportError:
      figure_type = None
    if figure_type is not None and isinstance(value, figure_type):
      buffer = io.BytesIO()
      savefig = getattr(value, "savefig")
      savefig(buffer, format="png")
      bundle["image/png"] = base64.b64encode(buffer.getvalue()).decode("ascii")
  return bundle


def _cell_error(exc):
  frames = [frame for frame in traceback.extract_tb(exc.__traceback__) if frame.filename == "<cell>"]
  # The terminal diagnostic is control, not another unbounded data frame.
  message = str(exc)
  if len(message) > 8192:
    message = message[:8192] + " [exception message truncated]"
  name = type(exc).__name__
  return {"ename": name, "evalue": message, "traceback": traceback.format_list(frames) + [f"{name}: {message}\n"]}


class _CellTransport:
  """A cell's immutable stdin/stdout adapter for the ONE shared hank client."""

  def __init__(self, cell_id, token, budget, send, origin, pending, pending_lock):
    self.cell_id = cell_id
    self.token = token
    self.budget = budget
    self._send = send
    self._origin = origin
    self._pending = pending
    self._pending_lock = pending_lock
    self.closed = False

  def exchange(self, payload):
    call_id = payload["call_id"]
    if not self.token:
      return {"call_id": call_id, "ok": False, "error": {"code": "bridge_unavailable", "message": "no bridge for this cell"}}
    if self.closed or self._origin() != self.cell_id:
      return {"call_id": call_id, "ok": False, "error": {"code": "bridge_closed", "message": "the originating cell has ended"}}
    waiter = {"event": threading.Event(), "reply": None, "transport": self}
    with self._pending_lock:
      if self.closed:
        raise OSError("the originating cell has ended")
      self._pending[call_id] = waiter
    try:
      self._send(payload, self.budget)
      waiter["event"].wait()
      if waiter["reply"] is None:
        raise OSError("the originating cell has ended")
      return waiter["reply"]
    finally:
      with self._pending_lock:
        self._pending.pop(call_id, None)

  def close(self):
    with self._pending_lock:
      self.closed = True
      for waiter in self._pending.values():
        if waiter["transport"] is self:
          waiter["event"].set()


def main():
  frame_limit = int(sys.argv[1]) if len(sys.argv) > 1 else MAX_FRAME_BYTES
  aggregate_limit = int(sys.argv[2]) if len(sys.argv) > 2 else MAX_CELL_BYTES
  signal.signal(signal.SIGINT, signal.SIG_IGN)
  control_input = sys.__stdin__
  native_stdout = sys.__stdout__
  native_stderr = sys.__stderr__
  assert control_input is not None
  assert native_stdout is not None
  assert native_stderr is not None
  frame_output = os.fdopen(os.dup(1), "wb", buffering=0)
  diagnostic_output = os.fdopen(os.dup(2), "wb", buffering=0)
  output_lock = threading.Lock()
  active_lock = threading.Lock()
  active: _ActiveCell = {"id": None, "budget": None, "fault": None}
  controls = queue.Queue()
  pending = {}
  pending_lock = threading.Lock()
  chunk_size = max(1, (frame_limit - 256) // 6)
  interrupt_state = {"sending": False, "deferred": False}

  def interrupt(signum, frame):
    if interrupt_state["sending"]:
      interrupt_state["deferred"] = True
    else:
      raise KeyboardInterrupt

  def send(frame, budget=None):
    on_main_thread = threading.current_thread() is threading.main_thread()
    if on_main_thread:
      interrupt_state["sending"] = True
    try:
      with output_lock:
        line = encode_frame(frame, budget)
        remaining = memoryview(line)
        while remaining:
          written = frame_output.write(remaining)
          remaining = remaining[written:]
    finally:
      if on_main_thread:
        interrupt_state["sending"] = False
        if interrupt_state["deferred"]:
          interrupt_state["deferred"] = False
          raise KeyboardInterrupt

  def emit(frame):
    with active_lock:
      budget = active["budget"]
    send(frame, budget)

  def diagnostic(message):
    diagnostic_output.write((message + "\n").encode("utf-8", errors="replace"))

  def record_fault(exc):
    with active_lock:
      if active["id"] is None:
        diagnostic(str(exc))
        return
      first = active["fault"] is None
      active["fault"] = exc
    if first and threading.current_thread() is not threading.main_thread():
      os.kill(os.getpid(), signal.SIGINT)

  def origin():
    thread = threading.current_thread()
    if thread is threading.main_thread():
      with active_lock:
        return active["id"]
    return getattr(thread, "_hank_origin_cell", None)

  # Private control and protocol descriptors are non-inheritable. Children get
  # /dev/null as stdin and the output capture pipes, never bridge descriptors.
  original_stdin = os.fdopen(os.dup(control_input.fileno()), "rb")
  null_fd = os.open(os.devnull, os.O_RDONLY)
  os.dup2(null_fd, 0)
  os.close(null_fd)
  sys.stdin = _NoInput()
  captures = {}
  for fd, name in ((1, "stdout"), (2, "stderr")):
    read_fd, write_fd = os.pipe()
    os.dup2(write_fd, fd)
    os.close(write_fd)
    os.set_blocking(read_fd, False)
    captures[read_fd] = (name, codecs.getincrementaldecoder("utf-8")(errors="replace"))
  wake_read, wake_write = os.pipe()
  os.set_blocking(wake_read, False)
  flushes = queue.Queue()
  buffers = {}

  def capture_text(name, text):
    with active_lock:
      cell_id = active["id"]
      failed = active["fault"] is not None
    if cell_id is not None and not failed:
      buffer = buffers.get(name)
      if buffer is None or buffer._cell_id != cell_id:
        buffer = _FrameBuffer(emit, cell_id, name, chunk_size)
        buffers[name] = buffer
      try:
        buffer.write(text)
      except ProtocolError as exc:
        record_fault(exc)

  def drain(fd):
    name, decoder = captures[fd]
    while True:
      try:
        chunk = os.read(fd, 8192)
      except BlockingIOError:
        return
      if not chunk:
        return
      capture_text(name, decoder.decode(chunk))

  def capture_output():
    try:
      while True:
        readable, _, _ = select.select([*captures, wake_read], [], [])
        for fd in readable:
          if fd != wake_read:
            drain(fd)
        if wake_read in readable:
          os.read(wake_read, 8192)
          # Snapshot requests BEFORE draining. A concurrent/cancelled flush
          # cannot acknowledge a later writer's bytes with an older barrier.
          requests = []
          while True:
            try:
              requests.append(flushes.get_nowait())
            except queue.Empty:
              break
          for fd in captures:
            drain(fd)
            name, decoder = captures[fd]
            capture_text(name, decoder.decode(b"", final=True))
            decoder.reset()
            buffer = buffers.get(name)
            if buffer is not None:
              try:
                buffer.flush()
              except ProtocolError as exc:
                record_fault(exc)
          for ready in requests:
            ready.set()
    except BaseException as exc:
      diagnostic(f"kernel output capture failed: {exc}")
      os._exit(1)

  def flush_capture():
    native_stdout.flush()
    native_stderr.flush()
    ready = threading.Event()
    flushes.put(ready)
    os.write(wake_write, b"\0")
    ready.wait()

  def read_input():
    try:
      while True:
        line = original_stdin.readline(frame_limit + 1)
        if not line:
          break
        try:
          frame = decode_frame(line)
        except ProtocolError as exc:
          # Drain the entire bad line without allocating an unbounded frame.
          while line and not line.endswith(b"\n"):
            line = original_stdin.readline(frame_limit + 1)
          record_fault(exc)
          continue
        if frame.get("type") in ("cell", "exit"):
          controls.put((frame, len(line)))
          continue
        with pending_lock:
          terminal_error = frame.get("call_id") is None and (frame.get("error") or {}).get("code") == "protocol_error"
          waiter = pending.get(frame.get("call_id"))
          # Exhausted budgets use a fixed-size terminal envelope, with no
          # correlation field. It must settle the waiting cell, not hang it.
          waiters = tuple(pending.values()) if terminal_error else (waiter,) if waiter is not None else ()
          for waiter in waiters:
            transport = waiter["transport"]
            try:
              if (frame.get("error") or {}).get("code") == "protocol_error":
                transport.budget.fail((frame.get("error") or {}).get("message") or "protocol_error")
              else:
                transport.budget.charge(len(line))
              waiter["reply"] = frame
            except ProtocolError as exc:
              waiter["reply"] = {"call_id": frame.get("call_id"), "ok": False, "error": {"code": "protocol_error", "message": str(exc)}}
            waiter["event"].set()
    except BaseException as exc:
      diagnostic(f"kernel input reader failed: {exc}")
    finally:
      with pending_lock:
        for waiter in pending.values():
          waiter["event"].set()
      controls.put(({"type": "exit"}, 0))

  parent_pid = os.getppid()

  def watch_parent():
    while True:
      time.sleep(1)
      if os.getppid() != parent_pid:
        os._exit(0)

  threading.Thread(target=capture_output, daemon=True).start()
  threading.Thread(target=read_input, daemon=True).start()
  threading.Thread(target=watch_parent, daemon=True).start()
  original_thread_init = threading.Thread.__init__

  def cell_thread_init(self, *args, **kwargs):
    original_thread_init(self, *args, **kwargs)
    # Capture origin at construction, including threads spawned by an older
    # background thread while a newer cell is running.
    self._hank_origin_cell = origin()

  threading.Thread.__init__ = cell_thread_init
  user_ns: dict[str, object] = {"__name__": "__main__"}
  while True:
    cell, wire_bytes = controls.get()
    if cell["type"] == "exit":
      frame_output.flush()
      diagnostic_output.flush()
      os._exit(0)
    cell_id = cell["id"]
    budget = CellBudget(frame_limit, aggregate_limit)
    transport = _CellTransport(cell_id, cell.get("cell_token"), budget, send, origin, pending, pending_lock)
    stdout = _Stream(cell_id, 1, origin, flush_capture)
    stderr = _Stream(cell_id, 2, origin, flush_capture)
    old_stdout, old_stderr = sys.stdout, sys.stderr
    env = cell.get("env") or {}
    old_env = {key: os.environ.get(key) for key in env}
    error = None
    cancelled = False
    with active_lock:
      active.update(id=cell_id, budget=budget, fault=None)

    def display(value, _cell_id=cell_id, _transport=transport):
      if _transport.closed or origin() != _cell_id:
        raise OSError("the originating cell has ended")
      flush_capture()
      emit({"type": "display", "id": _cell_id, "bundle": _mime_bundle(value)})

    user_ns.update(
      display=display,
      hank=_bridge_client.HankBridge(transport=transport, token=cell.get("cell_token")),
      HankBridgeError=_bridge_client.HankBridgeError,
      _hank_transport=transport,
      _hank_cell_token=cell.get("cell_token"),
      __file__=cell["script"],
    )
    try:
      sys.stdout, sys.stderr = stdout, stderr
      signal.signal(signal.SIGINT, interrupt)
      budget.charge(wire_bytes)
      emit({"type": "started", "id": cell_id})
      for key, value in env.items():
        if value is None:
          os.environ.pop(key, None)
        else:
          os.environ[key] = str(value)
      token = cell.get("cell_token")
      if token is not None:
        os.environ["AGENT_CODE_EXECUTE_CELL_TOKEN"] = token
      else:
        os.environ.pop("AGENT_CODE_EXECUTE_CELL_TOKEN", None)
      with open(cell["script"], encoding="utf-8") as script:
        code = script.read()
      # Only explicit display frames return values. A last hank.call expression
      # stays in the interpreter, not implicitly in the model's result.
      exec(compile(code, "<cell>", "exec"), user_ns)
    except KeyboardInterrupt as exc:
      with active_lock:
        fault = active["fault"]
      cancelled = fault is None
      error = _cell_error(fault or exc)
    except BaseException as exc:
      error = _cell_error(exc)
    finally:
      signal.signal(signal.SIGINT, signal.SIG_IGN)
      flush_capture()
      transport.close()
      sys.stdout, sys.stderr = old_stdout, old_stderr
      for key, value in old_env.items():
        if value is None:
          os.environ.pop(key, None)
        else:
          os.environ[key] = value
      os.environ.pop("AGENT_CODE_EXECUTE_CELL_TOKEN", None)
      with active_lock:
        fault = active["fault"]
        if fault is not None:
          error = _cell_error(fault)
          cancelled = False
        active.update(id=None, budget=None, fault=None)
    done = {"type": "done", "id": cell_id, "status": "error" if error or budget.error else "ok", "cancelled": cancelled}
    if error is not None:
      done["error"] = error
    if budget.error is not None:
      done["protocol_error"] = budget.error
    try:
      send(done, budget if budget.error is None else None)
    except ProtocolError as exc:
      done["status"] = "error"
      done["protocol_error"] = str(exc)
      send(done)


if __name__ == "__main__":
  sys.exit(main())
