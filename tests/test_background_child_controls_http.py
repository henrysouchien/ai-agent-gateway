from __future__ import annotations

import asyncio

import httpx


from agent_gateway import AgentSDKRunner, TaskRegistry, TaskState
from agent_gateway.event_log import EventLog
from hank_model_transport.policy_imports import configure_server_policy
from hank_model_transport.providers import AgentSDKConfig
from agent_gateway.session import SessionStream
from gateway_test_support.host_policy import owner_session_host_policy  # noqa: F401
from gateway_test_support.sdk_capability_execution_test_support import stub_sdk_capability_execution
from test_background_child_interrupt_resume import children  # noqa: F401


def test_user_interrupt_uses_only_authenticated_live_registry(
  make_test_app, children, owner_session_host_policy,
) -> None:
  async def case() -> None:
    app = make_test_app()
    configure_server_policy(owner_session_host_policy)
    live = children()
    parent = asyncio.current_task()
    assert parent is not None
    try:
      first = await live.start("first")
      sibling = await live.start("sibling")
      original = first.admitted_task
      assert original is not None
      assert first.asyncio_task is not None
      assert sibling.asyncio_task is not None
      async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test",
      ) as client:
        async def init():
          response = await client.post("/api/chat/init", json={
            "api_key": "gateway-key", "user_id": "alice",
          })
          assert response.status_code == 200
          return response.json()

        info = await init()
        other_info = await init()
        session = app.state.auth.session_store.get_session(info["session_id"])
        other = app.state.auth.session_store.get_session(other_info["session_id"])
        assert session is not None and other is not None
        session.active_turn = SessionStream(
          event_log=EventLog(session_id=session.session_id), runner_task=parent,
          runtime_owner=live.runner,
        )
        # Another authenticated conversation cannot control this registry.
        other.active_turn = SessionStream(
          event_log=EventLog(session_id=other.session_id), runner_task=parent,
          runtime_owner=type("OtherRunner", (), {
            "interrupt_background_task": lambda _self, task_id, *, reason: TaskRegistry().interrupt(task_id, reason=reason),
          })(),
        )
        url = f"/api/chat/background/{first.task_id}/interrupt"
        denied = await client.post(url)
        assert denied.status_code == 401
        foreign = await client.post(url, headers={
          "Authorization": f"Bearer {other_info['session_token']}",
        })
        assert foreign.status_code == 200
        assert foreign.json()["interrupt_requested"] is False
        assert first.interruption_reason is None
        response = await client.post(url, headers={
          "Authorization": f"Bearer {info['session_token']}",
        })
        assert response.status_code == 200
        assert response.json() == {"task_id": first.task_id, "interrupt_requested": True}
        await asyncio.wait_for(asyncio.shield(first.asyncio_task), timeout=5)
        assert first.state == TaskState.INTERRUPTED
        assert first.task_result is not None
        assert first.task_result.execution.terminal_reason == "user_interrupted"
        assert first.admitted_task == original
        assert sibling.state == TaskState.RUNNING
        assert sibling.asyncio_task.cancelling() == 0
        assert parent.cancelling() == 0
        assert session.active_turn.runtime_owner is live.runner
        result, error = await live.run_agent({"resume": first.task_id})
        assert error is None and result is not None
        resumed = live.runner._task_registry.get(result["task_id"])
        assert resumed is not None and resumed.asyncio_task is not None
        await asyncio.wait_for(asyncio.shield(resumed.asyncio_task), timeout=5)
        assert resumed.state == TaskState.COMPLETED
        assert resumed.admitted_task is not None
        assert resumed.admitted_task.model_bind == original.model_bind
        # These parent tasks belong to the test, not the app shutdown lifecycle.
        session.active_turn = None
        other.active_turn = None
    finally:
      await live.close()
      await app.state.user_event_bus.shutdown()

  asyncio.run(case())


def test_user_interrupt_without_live_runner_does_not_create_one(make_test_app) -> None:
  async def case() -> None:
    app = make_test_app()
    try:
      async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test",
      ) as client:
        init = await client.post("/api/chat/init", json={
          "api_key": "gateway-key", "user_id": "alice",
        })
        info = init.json()
        session = app.state.auth.session_store.get_session(info["session_id"])
        assert session is not None
        for turn in (None, SessionStream(event_log=EventLog(), runner_task=None)):
          session.active_turn = turn
          response = await client.post("/api/chat/background/unknown/interrupt", headers={
            "Authorization": f"Bearer {info['session_token']}",
          })
          assert response.status_code == 409
          assert session.active_turn is turn
    finally:
      await app.state.user_event_bus.shutdown()

  asyncio.run(case())


def test_user_interrupt_sdk_runner_returns_unsupported_without_cancelling_turn(
  make_test_app,
) -> None:
  async def case() -> None:
    app = make_test_app()
    parent = asyncio.current_task()
    assert parent is not None
    session = None
    try:
      async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test",
      ) as client:
        init = await client.post("/api/chat/init", json={
          "api_key": "gateway-key", "user_id": "alice",
        })
        assert init.status_code == 200
        info = init.json()
        session = app.state.auth.session_store.get_session(info["session_id"])
        assert session is not None
        event_log = EventLog(session_id=session.session_id)
        runner = AgentSDKRunner(
          event_log=event_log,
          session_id=session.session_id,
          sdk_config=AgentSDKConfig(
            user_id="alice", billing_mode="byok", rate_table_version="unknown",
          ),
          capability_execution=stub_sdk_capability_execution(),
          system_prompt="test",
        )
        turn = SessionStream(
          event_log=event_log, runner_task=parent, runtime_owner=runner,
        )
        session.active_turn = turn
        response = await client.post("/api/chat/background/child-sdk/interrupt", headers={
          "Authorization": f"Bearer {info['session_token']}",
        })
        assert response.status_code == 409
        detail = response.json()["detail"]
        assert detail["code"] == "background_child_interrupt_unsupported"
        assert detail["task_id"] == "child-sdk"
        assert isinstance(detail["message"], str) and detail["message"]
        assert session.active_turn is turn
        assert turn.runtime_owner is runner
        assert turn.runner_task is parent
        assert parent.cancelling() == 0
        assert event_log.entries == []
    finally:
      if session is not None:
        session.active_turn = None
      await app.state.user_event_bus.shutdown()

  asyncio.run(case())
