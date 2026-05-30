from __future__ import annotations

import asyncio
from contextlib import suppress
from typing import AsyncIterator

import pytest

from agent_gateway.event_log import UserEventBus


async def _next_event(iterator: AsyncIterator[dict], timeout: float = 0.5) -> dict:
  return await asyncio.wait_for(iterator.__anext__(), timeout=timeout)


async def _collect(iterator: AsyncIterator[dict], count: int) -> list[dict]:
  events = []
  for _ in range(count):
    events.append(await _next_event(iterator))
  return events


def test_user_event_bus_isolates_users() -> None:
  async def _run() -> None:
    bus = UserEventBus()
    alice = bus.subscribe("alice")
    bob = bus.subscribe("bob")
    alice_task = asyncio.create_task(alice.__anext__())
    bob_task = asyncio.create_task(bob.__anext__())
    await asyncio.sleep(0)

    await bus.publish("alice", "run-1", {"type": "message", "seq": 1})

    assert await asyncio.wait_for(alice_task, timeout=0.5) == {"type": "message", "seq": 1}
    with pytest.raises(asyncio.TimeoutError):
      await asyncio.wait_for(asyncio.shield(bob_task), timeout=0.05)

    bob_task.cancel()
    with suppress(asyncio.CancelledError):
      await bob_task
    await alice.aclose()
    await bob.aclose()
    await bus.shutdown()

  asyncio.run(_run())


def test_user_event_bus_preserves_order_within_run() -> None:
  async def _run() -> None:
    bus = UserEventBus()
    subscriber = bus.subscribe("alice", control_run_id="run-1")
    collector = asyncio.create_task(_collect(subscriber, 4))
    await asyncio.sleep(0)

    for seq in range(4):
      await bus.publish("alice", "run-1", {"type": "message", "seq": seq})
      await bus.publish("alice", "run-2", {"type": "message", "seq": 99})

    seen = await asyncio.wait_for(collector, timeout=0.5)
    assert [event["seq"] for event in seen] == [0, 1, 2, 3]

    await subscriber.aclose()
    await bus.shutdown()

  asyncio.run(_run())


def test_user_event_bus_backpressure_drops_oldest_and_emits_sentinel() -> None:
  async def _run() -> None:
    bus = UserEventBus(subscriber_queue_max=2)
    subscriber = bus.subscribe("alice", control_run_id="run-1")

    first_task = asyncio.create_task(subscriber.__anext__())
    await asyncio.sleep(0)
    await bus.publish("alice", "run-1", {"type": "message", "seq": 0})
    assert await asyncio.wait_for(first_task, timeout=0.5) == {"type": "message", "seq": 0}

    await bus.publish("alice", "run-1", {"type": "message", "seq": 1})
    await bus.publish("alice", "run-1", {"type": "message", "seq": 2})
    await bus.publish("alice", "run-1", {"type": "message", "seq": 3})

    sentinel = await _next_event(subscriber)
    assert sentinel["type"] == "events_dropped"
    assert sentinel["count"] == 1
    assert isinstance(sentinel["oldest_ts"], float)
    assert await _next_event(subscriber) == {"type": "message", "seq": 2}
    assert await _next_event(subscriber) == {"type": "message", "seq": 3}

    await subscriber.aclose()
    await bus.shutdown()

  asyncio.run(_run())


def test_user_event_bus_replays_buffer_before_live_tail() -> None:
  async def _run() -> None:
    bus = UserEventBus(replay_buffer_max=2)
    await bus.publish("alice", "run-1", {"type": "message", "seq": 1})
    await bus.publish("alice", "run-1", {"type": "message", "seq": 2})
    await bus.publish("alice", "run-1", {"type": "message", "seq": 3})

    subscriber = bus.subscribe("alice", control_run_id="run-1")
    assert await _next_event(subscriber) == {"type": "message", "seq": 2}
    assert await _next_event(subscriber) == {"type": "message", "seq": 3}

    live_task = asyncio.create_task(subscriber.__anext__())
    await asyncio.sleep(0)
    await bus.publish("alice", "run-1", {"type": "message", "seq": 4})
    assert await asyncio.wait_for(live_task, timeout=0.5) == {"type": "message", "seq": 4}

    await subscriber.aclose()
    await bus.shutdown()

  asyncio.run(_run())


def test_user_event_bus_fast_run_race_replays_after_termination_before_cleanup() -> None:
  async def _run() -> None:
    bus = UserEventBus()
    bus._cleanup_delay_seconds = 0.5
    await bus.publish("alice", "bg_1", {"type": "run_state_changed", "state": "running"})
    await bus.publish("alice", "bg_1", {"type": "run_state_changed", "state": "complete"})
    await bus.cleanup_run("alice", "bg_1")

    subscriber = bus.subscribe("alice", control_run_id="bg_1")
    assert await _next_event(subscriber) == {"type": "run_state_changed", "state": "running"}
    assert await _next_event(subscriber) == {"type": "run_state_changed", "state": "complete"}

    await subscriber.aclose()
    await bus.shutdown()

  asyncio.run(_run())


def test_user_event_bus_drops_replay_buffer_after_cleanup_delay() -> None:
  async def _run() -> None:
    bus = UserEventBus()
    bus._cleanup_delay_seconds = 0.01
    await bus.publish("alice", "run-1", {"type": "message", "seq": 1})
    await bus.cleanup_run("alice", "run-1")
    await asyncio.sleep(0.05)

    subscriber = bus.subscribe("alice", control_run_id="run-1")
    next_task = asyncio.create_task(subscriber.__anext__())
    await asyncio.sleep(0)
    with pytest.raises(asyncio.TimeoutError):
      await asyncio.wait_for(asyncio.shield(next_task), timeout=0.05)

    next_task.cancel()
    with suppress(asyncio.CancelledError):
      await next_task
    await subscriber.aclose()
    await bus.shutdown()

  asyncio.run(_run())


def test_user_event_bus_shutdown_and_cancelled_subscriber_cleanup_do_not_leak() -> None:
  async def _run() -> None:
    bus = UserEventBus()
    subscriber = bus.subscribe("alice", control_run_id="run-1")
    next_task = asyncio.create_task(subscriber.__anext__())
    await asyncio.sleep(0)

    shutdown_task = asyncio.create_task(bus.shutdown())
    next_task.cancel()
    with suppress(asyncio.CancelledError):
      await next_task
    await asyncio.wait_for(shutdown_task, timeout=0.5)

    assert bus._subscribers == {}
    assert bus._cleanup_tasks == {}

  asyncio.run(_run())
