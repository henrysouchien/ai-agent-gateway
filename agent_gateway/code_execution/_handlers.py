from __future__ import annotations

import os
import tempfile
import threading
import time
from dataclasses import dataclass, replace
from typing import Any, Callable, Dict, List, Optional

from ..agent_telemetry import make_prepare_env_with_agent_telemetry
from ..runner import ToolResultContext
from ..session import GatewaySession
from ..tool_dispatcher import ApprovalKeyQualifier, LocalToolHandler
from ._background import BackgroundTask, OutputRingBuffer
from ._backends import DockerBackend, ExecutionBackend, SubprocessBackend
from ._config import CodeExecutionConfig
from ._helpers import _boolean_input, _prepare_code_execute_env, _string_input, _timeout_ms_input, code_execute
from ._hooks import strip_code_execute_base64_hook
from ._tool_defs import make_code_execute_status_tool_def, make_code_execute_tool_def


@dataclass
class CodeExecutionBundle:
  """Bundle returned by `build_code_execution()`.

  It contains the local handlers, tool schemas, approval helpers, and result
  sanitization hook needed to plug code execution into a gateway runtime.
  """

  handlers: Dict[str, LocalToolHandler]
  tool_definitions: List[Dict[str, Any]]
  approval_qualifier: ApprovalKeyQualifier
  needs_approval: Callable[[str, Dict[str, Any] | None, str], bool]
  sanitize_hook: Callable[[ToolResultContext], None]
  ensure_work_dir: Callable[[], str]


def build_code_execution(
  session: GatewaySession,
  config: CodeExecutionConfig | None = None,
) -> CodeExecutionBundle:
  """Create built-in code execution tools for a session.

  The bundle prefers Docker when available and falls back to subprocess
  execution when registered. The session stores the persistent work directory and
  any background tasks created by `code_execute(background=true)`.
  """
  cfg = config or CodeExecutionConfig()

  backends: Dict[str, ExecutionBackend] = {}
  if cfg.register_subprocess:
    backends["subprocess"] = SubprocessBackend(config=cfg)
  if cfg.register_docker:
    backends["docker"] = DockerBackend(image=cfg.docker_image or None, config=cfg)

  def _get_backend(name: Optional[str] = None) -> ExecutionBackend:
    if name:
      backend = backends.get(name)
      if backend is not None:
        return backend
      raise ValueError(f"Unknown backend: '{name}'. Available: {sorted(backends)}")
    for preferred in ("docker", "subprocess"):
      backend = backends.get(preferred)
      if backend is not None and backend.available():
        return backend
    raise RuntimeError("No execution backend available")

  def _get_registered_backend_names() -> list[str]:
    return list(backends.keys())

  work_dir_lock = threading.Lock()

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

    per_call_config = replace(
      cfg,
      prepare_env=make_prepare_env_with_agent_telemetry(cfg.prepare_env, tool_ctx),
    )
    backend = _get_backend(resolved_host)
    if not backend.available():
      return None, {"code": "backend_unavailable", "message": f"Backend '{resolved_host}' unavailable"}

    work_dir = _ensure_code_execution_work_dir()
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

      handle = await backend.start(
        code,
        work_dir,
        task_id=task_id,
        timeout_ms=timeout_ms,
        env=env,
        on_output=_on_bg_output,
      )
      session.background_tasks[task_id] = BackgroundTask(
        task_id=task_id,
        handle=handle,
        backend=backend,
        stdout_buf=stdout_buf,
        stderr_buf=stderr_buf,
        started_at=time.time(),
      )
      return {
        "status": "running",
        "task_id": task_id,
        "message": "Use code_execute_status(task_id=...) to check progress.",
      }, None

    chunk_seq = [0]

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
          "text": text,
          "seq": chunk_seq[0],
        }
      )

    return await code_execute(
      tool_input,
      session_work_dir=work_dir,
      on_output=_on_chunk,
      backend=backend,
      config=per_call_config,
    )

  async def _handle_code_execute_status(tool_input: Dict[str, Any], **_: Any):
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
        return result, None
      await task.safe_cancel(backend)
      if task._terminated:
        session.background_tasks.pop(task_id, None)
      return {"status": "cancelled", "task_id": task_id}, None

    if poll_result.get("status") == "completed" or _handle_has_exited(task.handle):
      result = await task.safe_collect(backend)
      if task._terminated:
        session.background_tasks.pop(task_id, None)
      return result, None

    stdout_tail = task.stdout_buf.tail(20) or str(poll_result.get("stdout_tail") or "")
    stderr_tail = task.stderr_buf.tail(5) or str(poll_result.get("stderr_tail") or "")
    return {
      "status": "running",
      "task_id": task_id,
      "stdout_tail": stdout_tail,
      "stderr_tail": stderr_tail,
    }, None

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

  def _code_execute_input_will_fail_validation(tool_input: Dict[str, Any] | None) -> bool:
    if not isinstance(tool_input, dict):
      return True
    host, error = _string_input(tool_input, "host", default="auto")
    if error is not None or host not in {"auto", *set(_get_registered_backend_names())}:
      return True
    _, error = _boolean_input(tool_input, "background", default=False)
    if error is not None:
      return True
    _, error = _string_input(
      tool_input,
      "code",
      required_message="code is required",
      non_empty=True,
    )
    if error is not None:
      return True
    _, error = _timeout_ms_input(tool_input, cfg)
    return error is not None

  def _needs_approval(
    tool_name: str,
    tool_input: Dict[str, Any] | None = None,
    qualifier: str = "",
  ) -> bool:
    if tool_name != "code_execute":
      return False
    if _code_execute_input_will_fail_validation(tool_input):
      return False
    if qualifier:
      try:
        return not _get_backend(qualifier).sandboxed
      except (RuntimeError, ValueError):
        return True
    return True

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
    needs_approval=_needs_approval,
    sanitize_hook=strip_code_execute_base64_hook,
    ensure_work_dir=_ensure_code_execution_work_dir,
  )
