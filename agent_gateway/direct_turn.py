from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from dataclasses import dataclass
import inspect
from typing import Any, AsyncIterator, Mapping
from uuid import uuid4

from .event_log import EventLog
from .session import GatewaySession, SessionStream
from .runner_tool_execution import AgentRunnerDispatcher
from .server_chat_helpers import (
  active_turn_is_running,
  call_build_chat_runtime,
  prepare_session_driver_turn,
)
from .server_models import ChatTurnInputs


class DirectTurnBusyError(RuntimeError):
  def __init__(self) -> None:
    super().__init__("A turn is already running on this session")


@dataclass
class DirectTurnSession:
  session: GatewaySession
  runtime: Any
  runner: Any
  dispatcher: AgentRunnerDispatcher
  event_log: EventLog
  approval_store: Any
  prepared_turn: Any
  _durable_context_prepared: bool = False

  async def prepare_durable_context(self) -> None:
    if self._durable_context_prepared:
      return
    prepare = getattr(self.runner, "_prepare_run_durable_context", None)
    if callable(prepare) and getattr(self.runner, "_agent_session_log", None) is not None:
      pending = prepare(
        [],
        getattr(self.runtime, "system_prompt", None),
        resume_initial_messages=None,
        top_level_user_input=None,
        selected_content_committed=True,
        runner_id=f"runner_{uuid4().hex}",
      )
      if not inspect.isawaitable(pending):
        raise TypeError("durable run preparation returned a non-awaitable")
      await pending
    self._durable_context_prepared = True


@asynccontextmanager
async def open_direct_turn(
  session: GatewaySession,
  *,
  app_state: Any,
  request_id: str,
  context: Mapping[str, Any] | None = None,
  claim_active_turn: bool = True,
  prepare_durable_context: bool = False,
  prepare_turn_fn: Any = prepare_session_driver_turn,
  build_runtime_fn: Any = call_build_chat_runtime,
) -> AsyncIterator[DirectTurnSession]:
  if claim_active_turn and (
    active_turn_is_running(session.active_turn) or session.stream_active
  ):
    raise DirectTurnBusyError()

  event_log = EventLog(session_id=session.session_id, defer_terminal_close=True)
  active_turn = SessionStream(event_log=event_log, runner_task=asyncio.current_task())
  previous_turn = session.active_turn
  previous_stream_active = session.stream_active
  runtime = None
  runner = None
  try:
    turn_context = dict(context or {})
    turn_context["channel"] = session.channel
    prepared_turn = prepare_turn_fn(
      session,
      ChatTurnInputs(messages=[], request_id=request_id, context=turn_context,
                     metadata=None, model_key=None, claims_turn=claim_active_turn),
      build_chat_runtime=app_state.gateway_build_chat_runtime,
    )
    if claim_active_turn:
      if "dispatch_scope" in turn_context:
        session.dispatch_scope = turn_context["dispatch_scope"]
      active_turn.session = session
      session.active_turn = active_turn
      session.stream_active = True
      active_turn.attached.set()
    runtime = await build_runtime_fn(
      app_state.gateway_build_chat_runtime,
      session=session,
      request=prepared_turn.request,
      channel=prepared_turn.channel,
      auth_manager=app_state.auth,
      storage_root=app_state.autonomous_storage_root,
    )
    runner = runtime.build_runner(event_log, session.session_id, float(session.created_at))
    dispatcher = getattr(runner, "dispatcher", None) or getattr(runner, "_dispatcher", None)
    assert dispatcher is not None
    dispatcher.bind_research_file_activity_owner(runner.ensure_research_file_activity)
    if claim_active_turn:
      active_turn.runtime_owner = runner
      session.run_context = dispatcher.run_context
    direct_turn = DirectTurnSession(
      session=session, runtime=runtime, runner=runner, dispatcher=dispatcher,
      prepared_turn=prepared_turn, event_log=event_log,
      approval_store=(
        session._gateway_dispatcher_deps.approval_store
        if session._gateway_dispatcher_deps is not None else None
      ),
    )
    if prepare_durable_context:
      await direct_turn.prepare_durable_context()
    yield direct_turn
  finally:
    try:
      try:
        if runtime is not None:
          await runtime.on_disconnect()
      finally:
        try:
          if runner is not None:
            release_write = getattr(runner, "_release_write_lease", None)
            if callable(release_write):
              release_write()
        finally:
          try:
            if runner is not None:
              release_research = getattr(runner, "release_research_file_activity_lease_if_owned", None)
              if callable(release_research):
                release_research()
          finally:
            lease = active_turn.research_file_activity_lease
            active_turn.research_file_activity_lease = None
            if lease is not None:
              lease.release()
    finally:
      if claim_active_turn and session.active_turn is active_turn:
        session.active_turn = previous_turn
        session.stream_active = previous_stream_active
      active_turn.settled.set()
      event_log.close()
