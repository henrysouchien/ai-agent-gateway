from __future__ import annotations

import asyncio
import inspect
from contextvars import ContextVar, Token
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Generic, Protocol, TypeVar


class UiBlocksCapability(Protocol):
  """Structural type for the client contract capability pinned to a run."""

  contract_version: int
  manifest_digest: str


ResultT = TypeVar("ResultT")
CommitFn = Callable[[int, Callable[[], None]], Awaitable[ResultT] | ResultT]
PostRenameFailureFn = Callable[[BaseException, int], ResultT]


@dataclass(frozen=True)
class UiBlocksRunContext:
  """Run-scoped capability, identity, and idempotency state for UI blocks."""

  capability: UiBlocksCapability | None
  turn_key: str
  session_id: str
  registry: "UiBlocksRunRegistry"


_active_ui_blocks_run: ContextVar[UiBlocksRunContext | None] = ContextVar(
  "active_ui_blocks_run",
  default=None,
)


def activate_ui_blocks_run(context: UiBlocksRunContext) -> Token[UiBlocksRunContext | None]:
  """Make ``context`` available while the interactive runtime is constructed."""

  return _active_ui_blocks_run.set(context)


def reset_ui_blocks_run(token: Token[UiBlocksRunContext | None]) -> None:
  """Restore the prior runtime-construction carrier value."""

  _active_ui_blocks_run.reset(token)


def current_ui_blocks_run() -> UiBlocksRunContext | None:
  """Return the bound interactive run context, or ``None`` for detached paths."""

  return _active_ui_blocks_run.get()


class UiBlocksReservation(Generic[ResultT]):
  """One logical emission reservation returned by :meth:`reserve`.

  ``is_worker`` is true only for the first caller. Duplicates must call
  :meth:`wait`; it shields the shared Future so cancellation of one waiter
  cannot cancel the worker or other waiters. The worker must settle every exit:
  deterministic failures use :meth:`validation_failed`, cancellation outside
  commit uses :meth:`worker_cancelled`, and persistence/event work uses
  :meth:`commit`.

  ``commit_fn(index, mark_renamed)`` executes under the registry commit lock.
  It must call ``mark_renamed`` immediately after the atomic rename succeeds.
  That call permanently consumes the index. Exceptions before it remove the
  reservation; exceptions after it retain the committed identity and are
  converted with ``post_rename_failure``. Event append belongs after
  ``mark_renamed``; an append result of ``None`` must be raised as a post-rename
  failure rather than returned as acceptance.
  """

  def __init__(
    self,
    registry: "UiBlocksRunRegistry",
    ui_blocks_id: str,
    future: asyncio.Future[ResultT],
    *,
    is_worker: bool,
  ) -> None:
    self._registry = registry
    self.ui_blocks_id = ui_blocks_id
    self._future = future
    self.is_worker = is_worker

  async def wait(self) -> ResultT:
    """Await the shared outcome outside the lock without cancelling it."""

    if self.is_worker:
      raise RuntimeError("the reservation worker cannot wait on its own outcome")
    return await asyncio.shield(self._future)

  async def validation_failed(self, result: ResultT) -> ResultT:
    """Retain and publish a deterministic payload-specific failure."""

    self._require_worker()
    await self._registry._settle_result(self.ui_blocks_id, self._future, result)
    return result

  async def worker_cancelled(self, exc: asyncio.CancelledError) -> None:
    """Cancellation-safely unblock waiters and remove a pre-rename entry."""

    self._require_worker()
    await self._registry._settle_cancellation_safe(
      self.ui_blocks_id,
      self._future,
      exc,
      retain=False,
    )

  async def commit(
    self,
    commit_fn: CommitFn[ResultT],
    *,
    post_rename_failure: PostRenameFailureFn[ResultT],
  ) -> ResultT:
    """Allocate, persist, emit, and settle one accepted logical emission.

    The callback runs under the single registry lock. A successful return is
    retained as accepted. A pre-rename exception is published to current
    waiters and the entry is removed so a later call may retry. A post-rename
    exception consumes the index and resolves to the caller-provided structured
    internal-error result, which is retained for all later duplicates. Worker
    cancellation is settled before ``CancelledError`` is re-raised.
    """

    self._require_worker()
    return await self._registry._commit(
      self.ui_blocks_id,
      self._future,
      commit_fn,
      post_rename_failure=post_rename_failure,
    )

  def _require_worker(self) -> None:
    if not self.is_worker:
      raise RuntimeError("only the reservation worker may settle an outcome")


class UiBlocksRunRegistry:
  """Async-safe, run-wide two-phase registry for logical UI-block emissions.

  Callers derive ``ui_blocks_id`` before :meth:`reserve`. Exactly one worker
  performs validation and commit. Accepted results and deterministic
  validation failures remain in ``entries`` for the life of the run; retryable
  pre-rename infrastructure failures are removed. The accepted counter moves
  only when the commit callback signals successful atomic rename.
  """

  def __init__(self) -> None:
    self._lock = asyncio.Lock()
    self.entries: dict[str, asyncio.Future[Any]] = {}
    self._accepted_emission_count = 0

  @property
  def accepted_emission_count(self) -> int:
    return self._accepted_emission_count

  async def reserve(self, ui_blocks_id: str) -> UiBlocksReservation[Any]:
    """Reserve an id, returning a worker or a shielded duplicate handle."""

    async with self._lock:
      future = self.entries.get(ui_blocks_id)
      if future is not None:
        return UiBlocksReservation(self, ui_blocks_id, future, is_worker=False)
      future = asyncio.get_running_loop().create_future()
      self.entries[ui_blocks_id] = future
      return UiBlocksReservation(self, ui_blocks_id, future, is_worker=True)

  async def _settle_result(
    self,
    ui_blocks_id: str,
    future: asyncio.Future[ResultT],
    result: ResultT,
  ) -> None:
    async with self._lock:
      self._assert_current(ui_blocks_id, future)
      if not future.done():
        future.set_result(result)

  async def _settle_cancellation_safe(
    self,
    ui_blocks_id: str,
    future: asyncio.Future[Any],
    exc: asyncio.CancelledError,
    *,
    retain: bool,
  ) -> None:
    cleanup = asyncio.create_task(
      self._settle_exception(ui_blocks_id, future, exc, retain=retain)
    )
    while not cleanup.done():
      try:
        await asyncio.shield(cleanup)
      except asyncio.CancelledError:
        # Repeated cancellation may interrupt this waiter, but never cleanup.
        continue
    cleanup.result()

  async def _settle_exception(
    self,
    ui_blocks_id: str,
    future: asyncio.Future[Any],
    exc: BaseException,
    *,
    retain: bool,
  ) -> None:
    async with self._lock:
      self._settle_exception_locked(
        ui_blocks_id,
        future,
        exc,
        retain=retain,
      )

  def _settle_exception_locked(
    self,
    ui_blocks_id: str,
    future: asyncio.Future[Any],
    exc: BaseException,
    *,
    retain: bool,
  ) -> None:
    self._assert_current(ui_blocks_id, future)
    if not future.done():
      future.set_exception(exc)
      # The worker may have no duplicates. Mark the exception observed while
      # leaving it available to every present or future duplicate waiter.
      future.exception()
    if not retain:
      self.entries.pop(ui_blocks_id, None)

  async def _commit(
    self,
    ui_blocks_id: str,
    future: asyncio.Future[ResultT],
    commit_fn: CommitFn[ResultT],
    *,
    post_rename_failure: PostRenameFailureFn[ResultT],
  ) -> ResultT:
    async with self._lock:
      self._assert_current(ui_blocks_id, future)
      emission_index = self._accepted_emission_count
      renamed = False

      def mark_renamed() -> None:
        nonlocal renamed
        if renamed:
          raise RuntimeError("atomic rename was signalled more than once")
        renamed = True
        self._accepted_emission_count += 1

      try:
        outcome = commit_fn(emission_index, mark_renamed)
        if inspect.isawaitable(outcome):
          outcome = await outcome
        if not renamed:
          raise RuntimeError("commit callback returned before atomic rename")
      except asyncio.CancelledError as exc:
        self._settle_exception_locked(
          ui_blocks_id,
          future,
          exc,
          retain=renamed,
        )
        raise
      except BaseException as exc:
        if not renamed:
          self._settle_exception_locked(
            ui_blocks_id,
            future,
            exc,
            retain=False,
          )
          raise
        outcome = post_rename_failure(exc, emission_index)
        if not future.done():
          future.set_result(outcome)
        return outcome

      if not future.done():
        future.set_result(outcome)
      return outcome

  def _assert_current(
    self,
    ui_blocks_id: str,
    future: asyncio.Future[Any],
  ) -> None:
    if self.entries.get(ui_blocks_id) is not future:
      raise RuntimeError("UI-block reservation is no longer current")


__all__ = [
  "UiBlocksCapability",
  "UiBlocksReservation",
  "UiBlocksRunContext",
  "UiBlocksRunRegistry",
  "activate_ui_blocks_run",
  "current_ui_blocks_run",
  "reset_ui_blocks_run",
]
