from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from agent_gateway.direct_turn import DirectTurnBusyError, open_direct_turn


@pytest.fixture
def direct_setup():
  previous = SimpleNamespace(is_running=False)
  session = SimpleNamespace(
    session_id="direct-test", created_at=1, channel="test",
    active_turn=previous, stream_active=False,
    dispatch_scope={"previous": True}, run_context=object(),
  )
  dispatcher = SimpleNamespace(
    run_context=object(), approval_store=object(),
    bind_research_file_activity_owner=Mock(),
  )
  runner = SimpleNamespace(
    dispatcher=dispatcher, ensure_research_file_activity=Mock(),
    _release_write_lease=Mock(),
    release_research_file_activity_lease_if_owned=Mock(),
    _agent_session_log=object(), _prepare_run_durable_context=AsyncMock(),
  )
  session._gateway_dispatcher_deps = SimpleNamespace(approval_store=dispatcher.approval_store)
  runtime = SimpleNamespace(
    build_runner=Mock(return_value=runner), on_disconnect=AsyncMock(),
    system_prompt="system",
  )
  prepared = SimpleNamespace(request=object(), channel="test")
  state = SimpleNamespace(
    gateway_build_chat_runtime=object(), auth=object(), autonomous_storage_root=object(),
  )
  return SimpleNamespace(
    session=session, previous=previous, dispatcher=dispatcher, runner=runner,
    runtime=runtime, prepared=prepared, state=state,
    prepare=Mock(return_value=prepared), build=AsyncMock(return_value=runtime),
  )


def opening(setup, **kwargs):
  return open_direct_turn(
    setup.session, app_state=setup.state, request_id="request-test",
    prepare_turn_fn=setup.prepare, build_runtime_fn=setup.build, **kwargs,
  )


@pytest.mark.asyncio
async def test_claimed_turn_binds_and_restores(direct_setup):
  s = direct_setup
  async with opening(s, context={"dispatch_scope": {"new": True}}) as turn:
    active = s.session.active_turn
    assert active is not s.previous
    assert active.runtime_owner is s.runner
    assert active.attached.is_set()
    assert s.session.stream_active
    assert s.session.run_context is s.dispatcher.run_context
    assert s.session.dispatch_scope == {"new": True}
    assert turn.session is s.session
    assert turn.runtime is s.runtime
    assert turn.runner is s.runner
    assert turn.dispatcher is s.dispatcher
    assert turn.prepared_turn is s.prepared
    assert turn.approval_store is s.dispatcher.approval_store
    assert active.event_log is turn.event_log
    assert turn.event_log.defer_terminal_close
  assert s.session.active_turn is s.previous
  assert s.session.stream_active is False
  assert active.settled.is_set()
  s.runtime.on_disconnect.assert_awaited_once()
  s.runner._release_write_lease.assert_called_once()
  s.runner.release_research_file_activity_lease_if_owned.assert_called_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("busy_field", ["active", "stream"])
async def test_busy_refuses_before_preparation(direct_setup, busy_field):
  s = direct_setup
  if busy_field == "active":
    s.previous.is_running = True
  else:
    s.session.stream_active = True
  with pytest.raises(DirectTurnBusyError, match="^A turn is already running on this session$"):
    async with opening(s):
      pytest.fail("busy session admitted")
  s.prepare.assert_not_called()
  s.build.assert_not_called()
  assert s.session.active_turn is s.previous


@pytest.mark.asyncio
async def test_unclaimed_preserves_session_state_even_when_busy(direct_setup):
  s = direct_setup
  s.previous.is_running = True
  s.session.stream_active = True
  scope, run_context = s.session.dispatch_scope, s.session.run_context
  async with opening(s, claim_active_turn=False, context={"dispatch_scope": {"new": True}}):
    assert s.session.active_turn is s.previous
    assert s.session.stream_active is True
    assert s.session.dispatch_scope is scope
    assert s.session.run_context is run_context
  assert s.session.active_turn is s.previous
  assert s.session.stream_active is True
  assert s.session.dispatch_scope is scope
  assert s.session.run_context is run_context
  s.runtime.on_disconnect.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("claim", [True, False])
async def test_only_a_claimed_composition_asks_the_build_to_capture_the_turn(direct_setup, claim):
  s = direct_setup
  async with opening(s, claim_active_turn=claim):
    pass
  assert s.prepare.call_args.args[1].claims_turn is claim


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["prepare", "build", "runner", "body", "disconnect", "write", "research"])
async def test_failures_release_and_restore(direct_setup, failure):
  s = direct_setup
  failing = {
    "prepare": s.prepare, "build": s.build, "runner": s.runtime.build_runner,
    "disconnect": s.runtime.on_disconnect, "write": s.runner._release_write_lease,
    "research": s.runner.release_research_file_activity_lease_if_owned,
  }
  if failure in failing:
    failing[failure].side_effect = RuntimeError(failure)
  with pytest.raises(RuntimeError, match=failure):
    async with opening(s):
      if failure == "body":
        raise RuntimeError("body")
  assert s.session.active_turn is s.previous
  assert s.session.stream_active is False
  if failure not in {"prepare", "build"}:
    s.runtime.on_disconnect.assert_awaited_once()
  if failure not in {"prepare", "build", "runner"}:
    s.runner._release_write_lease.assert_called_once()
    s.runner.release_research_file_activity_lease_if_owned.assert_called_once()


@pytest.mark.asyncio
async def test_active_research_lease_released_after_disconnect_failure(direct_setup):
  s = direct_setup
  lease = SimpleNamespace(release=Mock())
  s.runtime.on_disconnect.side_effect = RuntimeError("disconnect")
  active = None
  with pytest.raises(RuntimeError, match="disconnect"):
    async with opening(s):
      active = s.session.active_turn
      active.research_file_activity_lease = lease
  lease.release.assert_called_once()
  assert active is not None
  assert active.research_file_activity_lease is None
  assert active.settled.is_set()


@pytest.mark.asyncio
@pytest.mark.parametrize("automatic", [True, False])
async def test_durable_preparation_once(direct_setup, automatic):
  s = direct_setup
  async with opening(s, prepare_durable_context=automatic) as turn:
    if not automatic:
      s.runner._prepare_run_durable_context.assert_not_awaited()
    await turn.prepare_durable_context()
    await turn.prepare_durable_context()
  s.runner._prepare_run_durable_context.assert_awaited_once()
  args, kwargs = s.runner._prepare_run_durable_context.call_args
  assert args == ([], "system")
  assert kwargs["selected_content_committed"] is True
  assert kwargs["runner_id"].startswith("runner_")


@pytest.mark.asyncio
async def test_durable_preparation_skips_missing_log(direct_setup):
  s = direct_setup
  s.runner._agent_session_log = None
  async with opening(s, prepare_durable_context=True):
    pass
  s.runner._prepare_run_durable_context.assert_not_called()


@pytest.mark.asyncio
async def test_durable_nonawaitable_releases_lifecycle(direct_setup):
  s = direct_setup
  s.runner._prepare_run_durable_context = Mock(return_value=None)
  with pytest.raises(TypeError, match="durable run preparation returned a non-awaitable"):
    async with opening(s, prepare_durable_context=True):
      pytest.fail("invalid durable preparation admitted")
  assert s.session.active_turn is s.previous
  s.runtime.on_disconnect.assert_awaited_once()
  s.runner._release_write_lease.assert_called_once()


@pytest.mark.asyncio
async def test_replacement_turn_is_not_overwritten(direct_setup):
  s = direct_setup
  replacement = object()
  async with opening(s) as turn:
    active = s.session.active_turn
    s.session.active_turn = replacement
    s.session.stream_active = True
  assert s.session.active_turn is replacement
  assert s.session.stream_active is True
  assert active.settled.is_set()
  assert turn.event_log.append({"type": "after_close"}) is None


@pytest.mark.asyncio
async def test_cancellation_releases_lifecycle(direct_setup):
  import asyncio

  s = direct_setup
  active = None
  turn = None
  with pytest.raises(asyncio.CancelledError):
    async with opening(s) as turn:
      active = s.session.active_turn
      raise asyncio.CancelledError()
  assert s.session.active_turn is s.previous
  assert s.session.stream_active is False
  assert active is not None
  assert turn is not None
  assert active.settled.is_set()
  assert turn.event_log.append({"type": "after_close"}) is None
  s.runtime.on_disconnect.assert_awaited_once()
  s.runner._release_write_lease.assert_called_once()
  s.runner.release_research_file_activity_lease_if_owned.assert_called_once()
