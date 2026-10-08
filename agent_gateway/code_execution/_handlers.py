from __future__ import annotations

import logging
import os
import tempfile
import threading
import time
from dataclasses import dataclass, replace
from itertools import islice
from typing import AbstractSet, Any, Callable, Dict, List, Optional

from ..agent_telemetry import make_prepare_env_with_agent_telemetry
from ..runner import ToolResultContext
from ..secret_boundary import SecretBoundary
from ..session import GatewaySession
from ..tool_dispatcher import ApprovalKeyQualifier, LocalToolHandler
from ..tool_policy_registry import PreparedToolCall, ToolInputPreparationError
from ._background import BackgroundTask, OutputRingBuffer
from ._backends import DockerBackend, ExecutionBackend, SubprocessBackend
from ._backends._kernel import KernelBackend, KernelState
from ._config import CodeExecutionConfig
from ._frame_reader import CatalogMetadata, CodeCallable, FrameReader, SourceExtractor, SourceKey
from ._helpers import _boolean_input, _prepare_code_execute_env, _string_input, _timeout_ms_input, code_execute
from ._hooks import strip_code_execute_base64_hook
from ._provenance import AGENT_CODE_EXECUTE_WORK_DIR_ENV, collect_computation_sidecars, delete_computation_sidecar_dir
from ._tool_defs import make_code_execute_status_tool_def, make_code_execute_tool_def

log = logging.getLogger(__name__)


@dataclass
class CodeExecutionBundle:
  """Bundle returned by `build_code_execution()`.

  It contains local handlers, tool schemas, approval helpers, a result hook
  and the session-owned `frame_reader` used by running cells. Request-scoped
  bundles share execution state, not dispatcher/configuration authority.
  """

  handlers: Dict[str, LocalToolHandler]
  tool_definitions: List[Dict[str, Any]]
  approval_qualifier: ApprovalKeyQualifier
  sanitize_hook: Callable[[ToolResultContext], None]
  ensure_work_dir: Callable[[], str]
  prepare_call: Callable[[Dict[str, Any]], PreparedToolCall]
  frame_reader: FrameReader

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
    """Bind the dispatcher (and its tool catalog) that nested cell calls go through.

    Each request's builder binds its own dispatcher; the latest bind wins, and
    a cell keeps the binding it opened under.
    `tool_definitions` is the turn's grant and `code_callable(operation)` the
    product registration's decision for that exact route; a nested call needs both.
    `extract_sources(operation, result)` is the product's per-tool source-handle
    extractor; its handles for successful nested calls become the outer
    result's `sources`, alongside the canonical dispatch settlement's sources.
    `source_key` is the citation registry's identity: equal keys are one source.
    `catalog_metadata` projects the installed route's producer identities;
    it neither resolves aliases nor grants an operation.
    """
    self.frame_reader.bind_dispatcher(
      dispatcher,
      tool_definitions=tool_definitions,
      extract_sources=extract_sources,
      source_key=source_key,
      code_callable=code_callable,
      catalog_metadata=catalog_metadata,
    )


def build_code_execution(
  session: GatewaySession,
  config: CodeExecutionConfig | None = None,
) -> CodeExecutionBundle:
  """Create built-in code execution tools for a session.

  `host=auto` prefers Docker, then a registered host kernel, then subprocess;
  skipped backend reasons are logged and returned with the selected backend.
  Reader, persistent interpreter states, work directory and background tasks
  belong to the session. Each bundle carries the current request's config.
  """
  cfg = config or CodeExecutionConfig()

  def _session_kernel(name: str) -> KernelState:
    state = session.code_execution_kernels.get(name)
    if state is None:
      state = KernelState()
      session.code_execution_kernels[name] = state
    return state

  backends: Dict[str, ExecutionBackend] = {}
  if cfg.register_subprocess:
    backends["subprocess"] = SubprocessBackend(config=cfg)
  if cfg.register_kernel:
    backends["kernel"] = KernelBackend(config=cfg, state=_session_kernel("kernel"))
  if cfg.register_docker:
    backends["docker"] = DockerBackend(
      image=cfg.docker_image or None, config=cfg, state=_session_kernel("docker"),
    )

  # `host=auto` takes the first backend in this order that can run. Each walk
  # records what it skipped and why, keyed by the backend it settled on, so the
  # call that then runs there reports the degradation it was selected under
  # (the handler receives only the resolved backend name).
  auto_skipped: Dict[str, List[Dict[str, Any]]] = {}

  def _get_backend(name: Optional[str] = None) -> ExecutionBackend:
    if name:
      backend = backends.get(name)
      if backend is not None:
        return backend
      raise ValueError(f"Unknown backend: '{name}'. Available: {sorted(backends)}")
    skipped: List[Dict[str, Any]] = []
    for preferred in ("docker", "kernel", "subprocess"):
      backend = backends.get(preferred)
      if backend is None:
        continue
      reason = backend.unavailable_reason()
      if reason is None:
        if skipped:
          log.warning(
            "code_execute host=auto selected %s (sandboxed=%s); skipped %s",
            backend.name,
            backend.sandboxed,
            "; ".join(f"{entry['backend']} (sandboxed={entry['sandboxed']}): {entry['reason']}" for entry in skipped),
          )
        auto_skipped[backend.name] = skipped
        return backend
      skipped.append({"backend": backend.name, "sandboxed": backend.sandboxed, "reason": reason})
    raise RuntimeError(
      "No execution backend available: "
      + "; ".join(f"{entry['backend']}: {entry['reason']}" for entry in skipped)
    )

  def _backend_identity(host: str, backend: ExecutionBackend) -> Dict[str, Any]:
    """What a code_execute result says about where it ran."""
    return {
      "backend": backend.name,
      "sandboxed": backend.sandboxed,
      "skipped_backends": list(auto_skipped.get(backend.name, ())) if host == "auto" else [],
    }

  def _get_registered_backend_names() -> list[str]:
    return list(backends.keys())

  work_dir_lock = threading.Lock()
  if session.code_execution_frame_reader is None:
    session.code_execution_frame_reader = FrameReader(principal={
      "user_id": session.user_id,
      "user_email": session.user_email,
      "risk_user_id": session.risk_user_id,
    })
  frame_reader = session.code_execution_frame_reader

  def _ensure_code_execution_work_dir() -> str:
    if session.code_execution_work_dir:
      return session.code_execution_work_dir
    with work_dir_lock:
      if not session.code_execution_work_dir:
        session.code_execution_work_dir = tempfile.mkdtemp(
          prefix=cfg.work_dir_prefix,
          dir=cfg.work_dir_root,
        )
    return session.code_execution_work_dir

  def _handle_has_exited(handle: Any) -> bool:
    backend_data = getattr(handle, "_backend_data", None)
    if not isinstance(backend_data, dict):
      return False
    process = backend_data.get("process")
    return process is not None and getattr(process, "returncode", None) is not None

  def _sandbox_work_dir_for_env(backend: ExecutionBackend, work_dir: str) -> str:
    if backend.name == "docker":
      return "/workspace"
    return work_dir

  def _output_redactor(tool_ctx: Any | None) -> Callable[[str], str]:
    # The dispatcher binds this one-string projection on the tool context
    # (`ToolExecutionContext.redact_text`); nothing else may answer to the name.
    redact: Callable[[str], str] | None = getattr(tool_ctx, "redact_text", None)
    if callable(redact):
      return redact
    return SecretBoundary().redact_text

  def _redact_execution_output(
    result: Any,
    redact: Callable[[str], str],
  ) -> Any:
    """Redact the child's own bytes before they leave this handler.

    A child can print or display a credential read from disk, and errors,
    sources and sidecars can carry those same bytes. Redact the whole
    model-visible result, including mapping keys, at this producer boundary.
    """
    # Admitted sidecars can be deeper than Python's call stack. Project them
    # post-order without recursion, preserving containers and mapping keys.
    pending: list[tuple[Any, bool]] = [(result, False)]
    values: list[Any] = []
    while pending:
      value, expanded = pending.pop()
      if isinstance(value, str):
        values.append(redact(value))
      elif isinstance(value, (list, tuple, dict)):
        if not expanded:
          pending.append((value, True))
          if isinstance(value, dict):
            for key, item in reversed(value.items()):
              pending.append((item, False))
              pending.append((key, False))
          else:
            pending.extend((item, False) for item in reversed(value))
          continue
        count = len(value) * 2 if isinstance(value, dict) else len(value)
        start = len(values) - count
        projected: Any
        if isinstance(value, dict):
          projected = {
            values[index]: values[index + 1]
            for index in range(start, len(values), 2)
          }
        elif isinstance(value, tuple):
          projected = tuple(islice(values, start, None))
        else:
          projected = values[start:]
        del values[start:]
        values.append(projected)
      else:
        values.append(value)
    return values[0]

  def _prepare_config_for_call(
    *,
    tool_ctx: Any | None,
    backend: ExecutionBackend,
    work_dir: str,
  ) -> CodeExecutionConfig:
    telemetry_prepare_env = make_prepare_env_with_agent_telemetry(cfg.prepare_env, tool_ctx)
    sandbox_work_dir = _sandbox_work_dir_for_env(backend, work_dir)

    def _prepare_env(env: Dict[str, str]) -> Dict[str, str]:
      prepared = telemetry_prepare_env(env)
      prepared[AGENT_CODE_EXECUTE_WORK_DIR_ENV] = sandbox_work_dir
      return prepared

    return replace(cfg, prepare_env=_prepare_env)

  async def _handle_code_execute(tool_input: Dict[str, Any], **kwargs: Any):
    host, error = _string_input(tool_input, "host", default="auto")
    if error is not None:
      return None, error
    assert host is not None
    valid_hosts = {"auto"} | set(_get_registered_backend_names())
    if host not in valid_hosts:
      return None, {"code": "invalid_input", "message": f"Unknown host: '{host}'"}

    tool_ctx = kwargs.get("tool_ctx")
    resolved_host = getattr(tool_ctx, "resolved_qualifier", "") if tool_ctx is not None else ""
    if not resolved_host:
      return None, {"code": "internal_error", "message": "Backend resolution failed"}
    # The outer call every nested `hank.call` of this cell is recorded under.
    outer_call_id = str(getattr(tool_ctx, "tool_call_id", "") or "")

    try:
      backend = _get_backend(resolved_host)
      backend_available = backend.available()
    except Exception:
      return None, {
        "code": "backend_unavailable",
        "message": f"Backend '{resolved_host}' unavailable",
      }
    if not backend_available:
      return None, {"code": "backend_unavailable", "message": f"Backend '{resolved_host}' unavailable"}

    work_dir = _ensure_code_execution_work_dir()
    per_call_config = _prepare_config_for_call(
      tool_ctx=tool_ctx,
      backend=backend,
      work_dir=work_dir,
    )
    background, error = _boolean_input(tool_input, "background", default=False)
    if error is not None:
      return None, error
    assert background is not None
    if background:
      code, error = _string_input(
        tool_input,
        "code",
        required_message="code is required",
        non_empty=True,
      )
      if error is not None:
        return None, error
      assert code is not None
      timeout_ms, error = _timeout_ms_input(tool_input, cfg)
      if error is not None:
        return None, error
      assert timeout_ms is not None
      env = _prepare_code_execute_env(per_call_config)
      task_id = f"ce_{os.urandom(4).hex()}"
      stdout_buf = OutputRingBuffer()
      stderr_buf = OutputRingBuffer()

      def _on_bg_output(stream_name: str, text: str) -> None:
        if stream_name == "stderr":
          stderr_buf.append(stream_name, text)
          return
        stdout_buf.append(stream_name, text)

      # A background cell outlives this turn; its own cancellation closes the
      # registration while the session's interpreter remains available.
      cell = frame_reader.open_cell(outer_call_id, abort_event=None, active=False)
      try:
        handle = await backend.start_bridged(
          code,
          work_dir,
          bridge=cell,
          task_id=task_id,
          timeout_ms=timeout_ms,
          env=env,
          on_output=_on_bg_output,
        )
      except BaseException:
        await cell.aclose()
        raise
      session.background_tasks[task_id] = BackgroundTask(
        task_id=task_id,
        handle=handle,
        backend=backend,
        stdout_buf=stdout_buf,
        stderr_buf=stderr_buf,
        started_at=time.time(),
        tool_call_id=getattr(tool_ctx, "tool_call_id", None),
        bridge=cell,
      )
      return {
        "status": "running",
        "task_id": task_id,
        "message": "Use code_execute_status(task_id=...) to check progress.",
        **_backend_identity(host, backend),
      }, None

    chunk_seq = [0]
    redact = _output_redactor(tool_ctx)

    def _on_chunk(stream_name: str, text: str) -> None:
      if tool_ctx is None:
        return
      chunk_seq[0] += 1
      tool_ctx.emit(
        {
          "type": "tool_output_chunk",
          "tool_call_id": tool_ctx.tool_call_id,
          "tool_name": "code_execute",
          "stream": stream_name,
          "text": redact(text),
          "seq": chunk_seq[0],
        }
      )

    cell = frame_reader.open_cell(
      outer_call_id,
      abort_event=getattr(tool_ctx, "abort_event", None),
      active=False,
    )
    try:
      result, error = await code_execute(
        tool_input,
        session_work_dir=work_dir,
        on_output=_on_chunk,
        backend=backend,
        config=per_call_config,
        bridge=cell,
      )
    finally:
      await cell.aclose()
    if result is not None:
      result.update(_backend_identity(host, backend))
      result.update(cell.summary())
    if result is not None and tool_ctx is not None:
      collect_computation_sidecars(
        result,
        work_dir=work_dir,
        tool_call_id=getattr(tool_ctx, "tool_call_id", None),
      )
    return _redact_execution_output(result, redact), error

  async def _handle_code_execute_status(tool_input: Dict[str, Any], **kwargs: Any):
    redact = _output_redactor(kwargs.get("tool_ctx"))
    task_id, error = _string_input(
      tool_input,
      "task_id",
      required_message="task_id is required",
      non_empty=True,
    )
    if error is not None:
      return None, error
    assert task_id is not None
    task_id = task_id.strip()
    cancel, error = _boolean_input(tool_input, "cancel", default=False)
    if error is not None:
      return None, error
    assert cancel is not None
    task = session.background_tasks.get(task_id)
    if task is None:
      return None, {"code": "not_found", "message": f"Unknown task_id: {task_id}"}

    backend = task.backend
    if task._in_progress:
      return {
        "status": "running",
        "task_id": task_id,
        "message": "Lifecycle op in progress",
      }, None

    poll_result = await backend.poll(task.handle)
    if cancel:
      if poll_result.get("status") == "completed" or _handle_has_exited(task.handle):
        result = await task.safe_collect(backend)
        if task._terminated:
          session.background_tasks.pop(task_id, None)
        if isinstance(result, dict):
          collect_computation_sidecars(
            result,
            work_dir=task.handle.work_dir,
            tool_call_id=task.tool_call_id,
          )
        return _redact_execution_output(result, redact), None
      await task.safe_cancel(backend)
      if task._terminated:
        session.background_tasks.pop(task_id, None)
      delete_computation_sidecar_dir(task.handle.work_dir, task.tool_call_id)
      return {"status": "cancelled", "task_id": task_id}, None

    if poll_result.get("status") == "completed" or _handle_has_exited(task.handle):
      result = await task.safe_collect(backend)
      if task._terminated:
        session.background_tasks.pop(task_id, None)
      if isinstance(result, dict):
        collect_computation_sidecars(
          result,
          work_dir=task.handle.work_dir,
          tool_call_id=task.tool_call_id,
        )
      return _redact_execution_output(result, redact), None

    stdout_tail = task.stdout_buf.tail(20) or str(poll_result.get("stdout_tail") or "")
    stderr_tail = task.stderr_buf.tail(5) or str(poll_result.get("stderr_tail") or "")
    return _redact_execution_output(
      {
        "status": "running",
        "task_id": task_id,
        "stdout_tail": stdout_tail,
        "stderr_tail": stderr_tail,
      },
      redact,
    ), None

  def _approval_qualifier(tool_name: str, tool_input: Dict[str, Any]) -> str:
    if tool_name != "code_execute":
      return ""
    host, error = _string_input(tool_input, "host", default="auto")
    if error is not None:
      return ""
    assert host is not None
    try:
      return _get_backend(host if host != "auto" else None).name
    except (RuntimeError, ValueError):
      return ""

  def _prepare_call(raw_input: Dict[str, Any]) -> PreparedToolCall:
    if not isinstance(raw_input, dict):
      raise ToolInputPreparationError({
        "code": "invalid_input",
        "message": "code_execute input must be an object",
      })
    prepared = dict(raw_input)
    host, error = _string_input(prepared, "host", default="auto")
    if error is not None:
      raise ToolInputPreparationError(error)
    assert host is not None
    valid_hosts = {"auto", *set(_get_registered_backend_names())}
    if host not in valid_hosts:
      raise ToolInputPreparationError({
        "code": "invalid_input",
        "message": f"Unknown host: '{host}'",
      })
    _, error = _boolean_input(prepared, "background", default=False)
    if error is not None:
      raise ToolInputPreparationError(error)
    _, error = _string_input(
      prepared,
      "code",
      required_message="code is required",
      non_empty=True,
    )
    if error is not None:
      raise ToolInputPreparationError(error)
    _, error = _timeout_ms_input(prepared, cfg)
    if error is not None:
      raise ToolInputPreparationError(error)
    try:
      backend = _get_backend(host if host != "auto" else None)
    except Exception as exc:
      raise ToolInputPreparationError({
        "code": "invalid_input",
        "message": "Code execution backend could not be resolved",
      }) from exc
    try:
      backend_available = host == "auto" or backend.available()
    except Exception as exc:
      raise ToolInputPreparationError({
        "code": "backend_unavailable",
        "message": "Code execution backend availability could not be determined",
      }) from exc
    if not backend_available:
      raise ToolInputPreparationError({
        "code": "backend_unavailable",
        "message": f"Backend '{backend.name}' unavailable",
      })
    return PreparedToolCall(prepared, backend.name)

  # No approval door on `code_execute` (it asked whenever the backend was
  # unsandboxed): posture decides whether the subprocess backend is registered
  # at all (`CodeExecutionConfig.register_subprocess`), so a registered
  # backend runs without a card.
  available_hosts = ("auto",) + tuple(_get_registered_backend_names())
  return CodeExecutionBundle(
    handlers={
      "code_execute": _handle_code_execute,
      "code_execute_status": _handle_code_execute_status,
    },
    tool_definitions=[
      make_code_execute_tool_def(cfg, available_hosts),
      make_code_execute_status_tool_def(),
    ],
    approval_qualifier=_approval_qualifier,
    sanitize_hook=strip_code_execute_base64_hook,
    ensure_work_dir=_ensure_code_execution_work_dir,
    prepare_call=_prepare_call,
    frame_reader=frame_reader,
  )


_docker_availability_cache: Dict[Optional[str], tuple[Optional[str], float]] = {}
_docker_availability_lock = threading.Lock()


def docker_backend_unavailable_reason(
  config: CodeExecutionConfig | None = None,
  *,
  ttl_s: float = 15.0,
) -> Optional[str]:
  """Why the Docker backend `config` names cannot run, or `None` when it can.

  Read-only: builds the same `DockerBackend` `build_code_execution` registers
  and returns its `unavailable_reason()` (`docker` on PATH plus a `docker image
  inspect` with a 3 s timeout). The answer is memoised per image name for
  `ttl_s` seconds so one turn does not pay the inspect twice.
  """
  cfg = config or CodeExecutionConfig()
  image = cfg.docker_image or None
  now = time.monotonic()
  with _docker_availability_lock:
    cached = _docker_availability_cache.get(image)
    if cached is not None and cached[1] > now:
      return cached[0]
  reason = DockerBackend(image=image, config=cfg).unavailable_reason()
  with _docker_availability_lock:
    _docker_availability_cache[image] = (reason, time.monotonic() + ttl_s)
  return reason


def clear_docker_availability_cache() -> None:
  """Forget every memoised Docker availability answer."""
  with _docker_availability_lock:
    _docker_availability_cache.clear()
