from __future__ import annotations

import asyncio
import logging
import shutil

from ..session import GatewaySession

log = logging.getLogger(__name__)


async def cleanup_code_execution(session: GatewaySession) -> None:
  """Release every session-owned code resource; a failed close cannot strand the rest.

  Teardown is resource containment: each failure is logged with its resource,
  and the remaining processes, pipes and work directory are still released.
  Cancellation is re-raised only after that release completes.

  A background cell is a `code` job on its runner's task registry; closing the
  session's frame reader kills and awaits the jobs of its open cells, so no
  cell runs in the work directory removed below.
  """
  cancelled: BaseException | None = None
  kernels = tuple(session.code_execution_kernels.values())
  session.code_execution_kernels.clear()
  reader = session.code_execution_frame_reader
  session.code_execution_frame_reader = None
  try:
    outcomes = await asyncio.gather(
      *(kernel.aclose() for kernel in kernels),
      *((reader.aclose(),) if reader is not None else ()),
      return_exceptions=True,
    )
    for outcome in outcomes:
      if isinstance(outcome, asyncio.CancelledError):
        cancelled = outcome
      elif isinstance(outcome, BaseException):
        log.warning("code_execute session resource close failed", exc_info=outcome)
  except asyncio.CancelledError as exc:
    cancelled = exc
  finally:
    if session.code_execution_work_dir:
      shutil.rmtree(session.code_execution_work_dir, ignore_errors=True)
      session.code_execution_work_dir = None
  if cancelled is not None:
    raise cancelled
