from __future__ import annotations

import shutil

from ..session import GatewaySession
from ._provenance import delete_computation_sidecar_dir


async def cleanup_code_execution(session: GatewaySession) -> None:
  """Cancel background code tasks and delete the session work directory."""
  for task in list(session.background_tasks.values()):
    try:
      await task.safe_cancel(task.backend)
    except Exception:
      pass
    try:
      delete_computation_sidecar_dir(task.handle.work_dir, getattr(task, "tool_call_id", None))
    except Exception:
      pass
  session.background_tasks.clear()
  if session.code_execution_work_dir:
    shutil.rmtree(session.code_execution_work_dir, ignore_errors=True)
    session.code_execution_work_dir = None
