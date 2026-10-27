"""Dependency-free persistent interpreter, shipped with the shared codec/client.

Control cells and bridge replies arrive on stdin. Stdout is exclusively bounded
NDJSON; Python, native and descendant output use separate capture pipes. The
same source runs locally and as Docker's long-lived interpreter. SIGINT ends a
cell, not the namespace; EOF/session disposal ends the interpreter.
"""
import ast
import asyncio
import base64
import codecs
import concurrent.futures
import importlib
import inspect
import io
import os
import queue
import select
import signal
import sys
import threading
import time
import types
import traceback
from typing import TypedDict

# Package imports use the source owners; the standalone launcher installs the
# same modules before entering this runner, without a gateway dependency.
if __package__:
  from . import _bridge_client, _frame_codec as _codec, _hank_prelude
  sys.modules.setdefault("_hank_bridge_client", _bridge_client)
  sys.modules.setdefault("_hank_frame_codec", _codec)
  sys.modules.setdefault("_hank_prelude", _hank_prelude)
else:
  _bridge_client = importlib.import_module("_hank_bridge_client")
  _codec = importlib.import_module("_hank_frame_codec")
  _hank_prelude = importlib.import_module("_hank_prelude")
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


def _implicit_mime_bundle(value):
  """Summarize a cell result without turning its size into execution failure.

  Only explicit display() invokes rich renderers. Implicit output is a single
  `text/plain` representation with a 16 KiB budget; the value itself stays
  untouched in the interpreter.
  """
  limit = 16 * 1024
  marker = "\n... [implicit result truncated at 16 KiB; full value is bound to `_`]"
  try:
    text = repr(value)
  except Exception as exc:
    text = f"<implicit repr unavailable: {type(exc).__name__}>"
  encoded = text[:limit].encode("utf-8", errors="backslashreplace")
  truncated = len(text) > limit or len(encoded) > limit
  if truncated:
    text = encoded[:limit - len(marker)].decode("utf-8", errors="ignore") + marker
  else:
    text = encoded.decode("utf-8")
  return {"text/plain": text}


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
    # Bridge requests this cell sent, and explicit display() calls it made;
    # the kernel-locals footer reads both after the cell ends.
    self.request_count = 0
    self.displays_emitted = 0

  def exchange(self, payload):
    """Send one bridge request and block on its reply.

    The gateway's compute clock is paused between `host_wait_begin` and
    `host_wait_end`, so a bracketed call means the cell is waiting on the host.
    The thread executing the cell (the main interpreter thread, which runs every
    `<cell>` frame) brackets its calls, and so does a `hank.acall` worker
    thread (`_HANK_PAUSES_CELL`). A call from any other cell-spawned thread is
    routed the same way but sends no frames, so its wait is compute. While a
    host wait is open the main thread may still run other coroutines; that CPU
    time also escapes `timeout_ms`. Begin precedes the request on the same
    ordered stdout, so a single long call is paused before the gateway starts
    dispatching it.
    """
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
      self.request_count += 1
    pauses_cell = threading.current_thread() is threading.main_thread() or bool(_bridge_client._HANK_PAUSES_CELL.get())
    try:
      if pauses_cell:
        self._send({"type": "host_wait_begin", "id": self.cell_id, "call_id": call_id}, self.budget)
      self._send(payload, self.budget)
      waiter["event"].wait()
      if waiter["reply"] is None:
        raise OSError("the originating cell has ended")
      return waiter["reply"]
    finally:
      with self._pending_lock:
        self._pending.pop(call_id, None)
      if pauses_cell:
        end = {"type": "host_wait_end", "id": self.cell_id, "call_id": call_id}
        # Like `done`, the end frame must reach the gateway on a poisoned budget,
        # or its compute clock would stay paused.
        try:
          self._send(end, self.budget)
        except ProtocolError:
          self._send(end)

  def close(self):
    with self._pending_lock:
      self.closed = True
      for waiter in self._pending.values():
        if waiter["transport"] is self:
          waiter["event"].set()


_SKIP_USER_NS_NAMES = frozenset({
  "__name__", "__doc__", "__package__", "__loader__", "__spec__",
  "__annotations__", "__builtins__", "__file__",
  "In", "Out", "display", "hank", "HankBridgeError",
  "_hank_transport", "_hank_cell_token", "_bind_existing",
})


def _summarize_user_ns(user_ns: dict[str, object]) -> dict[str, str]:
  """Compact live `name -> type/shape/keys` map of resident user variables."""
  summary: dict[str, str] = {}
  try:
    items = list(user_ns.items())
  except Exception:
    return summary
  for name, value in items:
    try:
      if not isinstance(name, str) or not name or name.startswith("_") or name in _SKIP_USER_NS_NAMES:
        continue
      if isinstance(value, types.ModuleType) or isinstance(value, type) or callable(value):
        continue
      bounded_name = name[:48]
      cls_name = type(value).__name__[:48]
      try:
        shape = getattr(value, "shape", None)
      except Exception:
        shape = None
      if cls_name == "DataFrame" and isinstance(shape, tuple) and len(shape) == 2:
        cols = [str(col)[:32] for col in list(getattr(value, "columns", ())[:6])]
        col_suffix = f"({','.join(cols)})" if cols else ""
        summary[bounded_name] = f"DataFrame[{shape[0]},{shape[1]}]{col_suffix}"
      elif isinstance(shape, tuple) and shape:
        summary[bounded_name] = f"{cls_name}[{','.join(str(dim)[:16] for dim in shape[:6])}]"
      elif isinstance(value, dict):
        keys: list[str] = []
        for k in value:
          keys.append(str(k)[:32])
          if len(keys) >= 6:
            break
        key_suffix = f"({','.join(keys)})" if keys else ""
        summary[bounded_name] = f"dict[{len(value)}]{key_suffix}"
      elif isinstance(value, (list, tuple, set)):
        summary[bounded_name] = f"{cls_name}[{len(value)}]"
      else:
        summary[bounded_name] = cls_name
      if len(summary) >= 40:
        break
    except Exception:
      continue
  return summary


def _prelude_binding_names(tree: ast.Module) -> list[str]:
  """Names bound by top-level `Assign` / `AnnAssign` statements, in order.

  Imports, defs and classes are deliberately absent: the prelude's helpers stay
  in `prelude_ns`, and only the plain bindings reach the user's namespace.
  """
  names: list[str] = []
  for node in tree.body:
    if isinstance(node, ast.Assign):
      targets = node.targets
    elif isinstance(node, ast.AnnAssign) and node.value is not None:
      targets = [node.target]
    else:
      continue
    for target in targets:
      for part in ast.walk(target):
        if isinstance(part, ast.Name) and isinstance(part.ctx, ast.Store) and part.id not in names:
          names.append(part.id)
  return names


def _leading_loads(code: str) -> tuple[str, list[tuple[int, str]]]:
  """Blank each leading `%load <path>` line and return `(code, [(line_no, path)])`.

  Only the lines before the first non-blank, non-`%load` line are magics. The
  `%load` line becomes an empty line, so `<cell>` line numbers stay those of the
  agent's code.
  """
  lines = code.split("\n")
  loads: list[tuple[int, str]] = []
  for index, line in enumerate(lines):
    stripped = line.strip()
    if not stripped:
      continue
    if stripped != "%load" and not (stripped.startswith("%load") and stripped[5].isspace()):
      break
    loads.append((index + 1, stripped[5:].strip()))
    lines[index] = ""
  return "\n".join(lines), loads


def _raise_at_cell_line(exc: BaseException, line_no: int):
  """Raise `exc` from a `<cell>` frame at `line_no`, where its `%load` line is."""
  frame = compile("\n" * (line_no - 1) + "raise _hank_cell_error", "<cell>", "exec")
  eval(frame, {"__builtins__": __builtins__, "_hank_cell_error": exc})


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

  def send(frame, budget=None, *, limit=MAX_CELL_BYTES):
    on_main_thread = threading.current_thread() is threading.main_thread()
    if on_main_thread:
      interrupt_state["sending"] = True
    try:
      with output_lock:
        line = encode_frame(frame, budget, limit=limit)
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

  def emit(frame, *, limit=MAX_CELL_BYTES):
    with active_lock:
      budget = active["budget"]
    send(frame, budget, limit=limit)

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
  # Bytes the running cell wrote to stdout (Python and native alike); the
  # capture thread is the one place every stdout byte passes.
  cell_output = {"stdout_bytes": 0}

  def capture_text(name, text):
    with active_lock:
      cell_id = active["id"]
      failed = active["fault"] is not None
    if cell_id is not None and name == "stdout":
      cell_output["stdout_bytes"] += len(text.encode("utf-8"))
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
        line = original_stdin.readline(MAX_CELL_BYTES + 1)
        if not line:
          break
        try:
          frame = decode_frame(line, limit=MAX_CELL_BYTES)
        except ProtocolError as exc:
          # Drain the entire bad line without allocating an unbounded frame.
          while line and not line.endswith(b"\n"):
            line = original_stdin.readline(MAX_CELL_BYTES + 1)
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
              # A correlated protocol_error reply settles only its waiter; it
              # must not poison the cell budget (the cell may catch it). Only
              # the uncorrelated terminal envelope fails the budget.
              if terminal_error and (frame.get("error") or {}).get("code") == "protocol_error":
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
  hank_bridge = _bridge_client.HankBridge()

  class _HankModule(types.ModuleType):
    # `import hank` must yield the bridge's surface. A real module object
    # delegates attribute access to the one HankBridge, so the injected
    # global and the imported module are the same object in cells.
    def __getattr__(self, name: str) -> object:
      return getattr(hank_bridge, name)

  hank_module = _HankModule("hank")
  user_ns["hank"] = hank_module
  user_ns["HankBridgeError"] = _bridge_client.HankBridgeError
  _hank_prelude.install_plot_capture()
  sys.modules["hank"] = hank_module
  # The product prelude runs here, not in `user_ns`: its imports, classes and
  # helpers stay out of the agent's globals; only its plain bindings are copied.
  prelude_ns: dict[str, object] = {
    "__name__": "_hank_prelude_ns",
    "hank": hank_module,
    "HankBridgeError": _bridge_client.HankBridgeError,
  }
  last_prelude_code: str | None = None
  # One loop carries the interpreter's coroutine functions and loop-bound
  # objects across cells. Tasks a cell leaves pending do not outlive it:
  # `settle_cell_loop` cancels them when the cell ends. The loop runs only
  # inside `run_compiled`, on the main thread, so no second thread ever drives
  # it; `settle_cell_loop` replaces it when a task will not cancel.
  cell_loop = asyncio.new_event_loop()
  settle_bound_s = 1.0

  def run_compiled(*code_objects):
    # eval() runs exec- and eval-mode code objects alike; the last one's value
    # is returned. The compiler's coroutine flag (not a result's type, so a
    # value that is a coroutine object the cell never awaited is not run) says a
    # part uses top-level `await`, `async for` or `async with`. Then every part
    # of the cell runs inside ONE coroutine driven on `cell_loop`, so plain
    # statements such as `asyncio.create_task(...)` find a running loop even
    # when the only `await` is in the trailing expression. A cell with no such
    # part runs as plain `eval`s with no loop involved.
    if not any(code.co_flags & inspect.CO_COROUTINE for code in code_objects):
      value = None
      for code in code_objects:
        value = eval(code, user_ns)
      return value

    async def drive():
      value = None
      for code in code_objects:
        value = eval(code, user_ns)
        if code.co_flags & inspect.CO_COROUTINE:
          value = await value
      return value

    return cell_loop.run_until_complete(drive())

  async def _cancel_tasks(tasks):
    # `wait`, not `gather`: a timeout must leave a task that swallows
    # cancellation running, not wait for it as cancelling a gather would.
    done, pending = await asyncio.wait(tasks, timeout=settle_bound_s)
    for task in done:
      if not task.cancelled():
        task.exception()
    return pending

  def settle_cell_loop(executor):
    # A cell ends with its tasks: one the cell left pending, or an interrupted
    # run left suspended, must not resume under a later cell's transport.
    # Cancellation handlers are user code running with SIGINT ignored, so the
    # wait is bounded: a task that will not end is abandoned with its loop, and
    # successor cells run on a fresh one. The namespace is untouched.
    nonlocal cell_loop
    tasks = asyncio.all_tasks(cell_loop)
    for task in tasks:
      task.cancel()
    stuck = cell_loop.run_until_complete(_cancel_tasks(tasks)) if tasks else ()
    executor.shutdown(wait=False, cancel_futures=True)
    if stuck:
      abandoned, cell_loop = cell_loop, asyncio.new_event_loop()
      # Destroying the abandoned pending tasks would log through the loop.
      abandoned.set_exception_handler(lambda loop, context: None)
      abandoned.close()
      diagnostic(f"{len(stuck)} task(s) ignored cancellation for {settle_bound_s:g}s; their event loop was abandoned")

  def run_load(path, line_no):
    # A `%load` failure surfaces as a cell error pointing at its own line.
    try:
      if not path:
        raise ValueError("%load requires a file path")
      if len(path) >= 2 and path[0] == path[-1] and path[0] in "\"'":
        path = path[1:-1]
      path = os.path.expanduser(path)
      with open(path, encoding="utf-8") as handle:
        source = handle.read()
      run_compiled(compile(source, path, "exec", flags=ast.PyCF_ALLOW_TOP_LEVEL_AWAIT))
    except Exception as exc:
      _raise_at_cell_line(exc, line_no)
  while True:
    cell, wire_bytes = controls.get()
    if cell["type"] == "exit":
      frame_output.flush()
      diagnostic_output.flush()
      os._exit(0)
    cell_id = cell["id"]
    budget = CellBudget(aggregate_limit=aggregate_limit)
    transport = _CellTransport(cell_id, cell.get("cell_token"), budget, send, origin, pending, pending_lock)
    stdout = _Stream(cell_id, 1, origin, flush_capture)
    stderr = _Stream(cell_id, 2, origin, flush_capture)
    # `asyncio.to_thread` workers inherit the origin of the cell that spawned
    # them (see `cell_thread_init`), so each cell gets its own default executor.
    cell_executor = concurrent.futures.ThreadPoolExecutor(thread_name_prefix="hank-acall")
    cell_loop.set_default_executor(cell_executor)
    old_stdout, old_stderr = sys.stdout, sys.stderr
    env = cell.get("env") or {}
    old_env = {key: os.environ.get(key) for key in env}
    error = None
    after_locals = None
    before_locals: dict[str, str] = {}
    cancelled = False
    cell_output["stdout_bytes"] = 0
    with active_lock:
      active.update(id=cell_id, budget=budget, fault=None)

    def display(value, _cell_id=cell_id, _transport=transport):
      if _transport.closed or origin() != _cell_id:
        raise OSError("the originating cell has ended")
      flush_capture()
      emit({"type": "display", "id": _cell_id, "bundle": _mime_bundle(value)}, limit=MAX_FRAME_BYTES)
      _transport.displays_emitted += 1

    hank_bridge._bind_cell(
      transport=transport,
      token=cell.get("cell_token"),
      parent_call_id=env.get("AGENT_TELEMETRY_TOOL_CALL_ID") or os.environ.get("AGENT_TELEMETRY_TOOL_CALL_ID"),
    )
    user_ns["hank"] = hank_module
    user_ns.update(
      display=display,
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
      # The prelude runs only now: `_RiskClient()` and friends read the cell's
      # env and `AGENT_CODE_EXECUTE_CELL_TOKEN` when constructed.
      _hank_prelude.set_plot_prefix(cell.get("plot_prefix") or "_plot_", os.getcwd())
      prelude_code = cell.get("prelude") or ""
      if prelude_code != last_prelude_code:
        exec(compile(prelude_code, "<hank_prelude>", "exec"), prelude_ns)
        for name in _prelude_binding_names(ast.parse(prelude_code)):
          if name in prelude_ns:
            user_ns[name] = prelude_ns[name]
        last_prelude_code = prelude_code
      with open(cell["script"], encoding="utf-8") as script:
        code = script.read()
      before_locals = _summarize_user_ns(user_ns)
      code, loads = _leading_loads(code)
      for line_no, path in loads:
        run_load(path, line_no)
      # Implicit output is a bounded summary, unlike explicit display().
      # `print(...)` remains stdout and a None result produces no bundle.
      tree = ast.parse(code)
      if tree.body and isinstance(tree.body[-1], ast.Expr):
        last_value = run_compiled(
          compile(
            ast.Module(body=tree.body[:-1], type_ignores=[]), "<cell>", "exec",
            flags=ast.PyCF_ALLOW_TOP_LEVEL_AWAIT,
          ),
          compile(
            ast.Expression(tree.body[-1].value), "<cell>", "eval",
            flags=ast.PyCF_ALLOW_TOP_LEVEL_AWAIT,
          ),
        )
        if last_value is not None:
          user_ns["_"] = last_value
          flush_capture()
          emit({"type": "display", "id": cell_id, "bundle": _implicit_mime_bundle(last_value)})
      else:
        run_compiled(compile(tree, "<cell>", "exec", flags=ast.PyCF_ALLOW_TOP_LEVEL_AWAIT))
      # Reaching here means the cell raised nothing. A cell that called the
      # bridge and printed nothing otherwise returns no trace of what it bound.
      flush_capture()
      after_locals = _summarize_user_ns(user_ns)
      if transport.request_count >= 1 and cell_output["stdout_bytes"] == 0 and transport.displays_emitted == 0:
        changed = [f"{name} = {shape}" for name, shape in after_locals.items() if before_locals.get(name) != shape]
        if changed:
          sys.stdout.write(f"[bound in kernel: {'; '.join(changed[:3])}]\n")
          flush_capture()
    except KeyboardInterrupt as exc:
      with active_lock:
        fault = active["fault"]
      cancelled = fault is None
      error = _cell_error(fault or exc)
    except BaseException as exc:
      error = _cell_error(exc)
    finally:
      signal.signal(signal.SIGINT, signal.SIG_IGN)
      try:
        settle_cell_loop(cell_executor)
      except Exception as exc:
        diagnostic(f"cell event loop cleanup failed: {exc}")
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
    done["kernel_locals"] = after_locals if after_locals is not None else _summarize_user_ns(user_ns)
    try:
      send(done, budget if budget.error is None else None)
    except ProtocolError as exc:
      done.pop("kernel_locals", None)
      done["status"] = "error"
      done["protocol_error"] = str(exc)
      send(done)


if __name__ == "__main__":
  sys.exit(main())
