"""Local tool handlers that act as one executing job.

A job-scoped handler (mailbox send, handle reads) admits its request as the
job its :class:`JobScope` names, and a spill read looks only under that job's
spill sink. A child's dispatcher starts from its parent's handler map, so each
such handler is rebuilt for the child's own scope with
:func:`rebind_job_handlers`; a copied closure never carries the parent's
authority or spill root into a child.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, MutableMapping
from dataclasses import dataclass
from typing import Any

from .admission import ParentContext

CallerContext = Callable[[], ParentContext]
LocalHandler = Callable[..., Awaitable[tuple[Any, Any]]]


@dataclass(frozen=True, slots=True)
class JobScope:
  """The executing job a handler acts as.

  ``spill_sink`` returns that job's own spill sink (``None``: the runtime
  has none). It is part of the scope because spill membership is location:
  a reader's own root and the job roots below it are its outputs.
  ``runner`` resolves the executing runner after child construction; ``session``
  is that job's session, so nested spawns use its authority and concurrency slot.
  """

  caller: CallerContext
  spill_sink: Callable[[], Any | None]
  runner: Callable[[], Any] = lambda: None
  session: Any | None = None


class JobScopedHandler:
  """A local handler built for one job's scope, rebuildable for another."""

  def __init__(
    self,
    build: Callable[[JobScope], LocalHandler],
    scope: JobScope,
  ) -> None:
    self._build = build
    self._handler = build(scope)

  async def __call__(self, tool_input: Any, **kwargs: Any) -> tuple[Any, Any]:
    return await self._handler(tool_input, **kwargs)

  def for_job(self, scope: JobScope) -> "JobScopedHandler":
    return JobScopedHandler(self._build, scope)


def rebind_job_handlers(
  handlers: MutableMapping[str, Any],
  scope: JobScope,
) -> None:
  """Rebuild every job-scoped handler in ``handlers`` for ``scope``."""

  for name, handler in list(handlers.items()):
    if isinstance(handler, JobScopedHandler):
      handlers[name] = handler.for_job(scope)


__all__ = [
  "CallerContext",
  "JobScope",
  "JobScopedHandler",
  "rebind_job_handlers",
]
