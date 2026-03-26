from __future__ import annotations

import shutil

from ..session import Session


async def cleanup_code_execution(session: Session) -> None:
  for task in list(session.background_tasks.values()):
    try:
      await task.safe_cancel(task.backend)
    except Exception:
      pass
  session.background_tasks.clear()
  if session.code_execution_work_dir:
    shutil.rmtree(session.code_execution_work_dir, ignore_errors=True)
    session.code_execution_work_dir = None
