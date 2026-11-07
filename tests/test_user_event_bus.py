from __future__ import annotations

import asyncio
from contextlib import suppress
from collections.abc import AsyncGenerator, AsyncIterator

import pytest

from agent_gateway.event_log import UserEventBus


def _subscribe(
  bus: UserEventBus,
  user_id: str,
  *,
  control_run_id: str | None = None,
) -> AsyncGenerator[dict, None]:
  iterator = bus.subscribe(user_id, control_run_id=control_run_id)
  assert isinstance(iterator, AsyncGenerator)
  return iterator


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
    alice = _subscribe(bus, "alice")
    bob = _subscribe(bus, "bob")
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
    subscriber = _subscribe(bus, "alice", control_run_id="run-1")
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
  """A subscriber nobody reads (stalled): once the producer's bounded wait is over
  the bus drops the oldest event and the sentinel names what was lost."""
  async def _run() -> None:
    bus = UserEventBus(subscriber_queue_max=2, backpressure_wait_seconds=0)
    subscriber = _subscribe(bus, "alice", control_run_id="run-1")

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


def test_a_live_global_watcher_slower_than_the_producers_loses_nothing() -> None:
  """R28 R10-04 (DEFECT-DESK 7f6cc54c): `control watch` over N streaming runs
  printed events_dropped on 95 % of lines. A global subscriber received every
  run's deltas into one run's worth of slots, and producers never waited for a
  live consumer, so drop-oldest discarded the stream. Now the bound scales with
  the runs a global subscriber spans and a producer that finds a live subscriber
  past half its bound waits for it to drain; a consumer slower than the producers
  still receives every event, in order, with no sentinel."""
  async def _run() -> None:
    n_runs, per_run = 5, 400
    bus = UserEventBus(subscriber_queue_max=100)
    subscriber = _subscribe(bus, "alice")  # global: every run of alice
    received: list[tuple[str, int]] = []
    dropped = 0

    async def consume() -> None:
      nonlocal dropped
      while len(received) + dropped < n_runs * per_run:
        event = await _next_event(subscriber)
        if event["type"] == "events_dropped":
          dropped += event["count"]
          continue
        received.append((event["run"], event["i"]))
        await asyncio.sleep(0.0002)  # a transport write per event: slower than the producers

    async def produce(run_id: str) -> None:
      for i in range(per_run):
        await bus.publish("alice", run_id, {"type": "text_delta", "run": run_id, "i": i})

    consumer = asyncio.create_task(consume())
    await asyncio.gather(*(produce(f"run-{r}") for r in range(n_runs)))
    await asyncio.wait_for(consumer, timeout=30)

    assert dropped == 0
    assert len(received) == n_runs * per_run
    for r in range(n_runs):
      assert [i for run, i in received if run == f"run-{r}"] == list(range(per_run))

    await subscriber.aclose()
    await bus.shutdown()

  asyncio.run(_run())


def test_a_stalled_subscriber_releases_the_producer_at_the_deadline() -> None:
  """Backpressure is bounded: a subscriber nobody reads holds a producer for at
  most backpressure_wait_seconds, then the bus drops as before. Closing the
  subscriber releases a waiting producer at once."""
  async def _run() -> None:
    # Bound 4: a producer waits once two events sit unread (half the bound).
    bus = UserEventBus(subscriber_queue_max=4, backpressure_wait_seconds=5.0)
    subscriber = _subscribe(bus, "alice", control_run_id="run-1")
    first_task = asyncio.create_task(subscriber.__anext__())  # registers the subscriber
    await asyncio.sleep(0)
    await bus.publish("alice", "run-1", {"type": "message", "seq": 0})
    assert await asyncio.wait_for(first_task, timeout=0.5) == {"type": "message", "seq": 0}

    # Nobody reads now: the second unread publish reaches the mark, so that
    # producer waits on the subscriber rather than finishing at once.
    await bus.publish("alice", "run-1", {"type": "message", "seq": 1})
    waiting = asyncio.create_task(bus.publish("alice", "run-1", {"type": "message", "seq": 2}))
    for _ in range(20):
      await asyncio.sleep(0)
    assert not waiting.done()

    # A read below the mark releases it (the live-consumer path).
    assert await _next_event(subscriber) == {"type": "message", "seq": 1}
    await asyncio.wait_for(waiting, timeout=1.0)

    # Closing the subscriber releases a waiting producer at once, deadline or not.
    assert await _next_event(subscriber) == {"type": "message", "seq": 2}
    await bus.publish("alice", "run-1", {"type": "message", "seq": 3})
    waiting = asyncio.create_task(bus.publish("alice", "run-1", {"type": "message", "seq": 4}))
    for _ in range(20):
      await asyncio.sleep(0)
    assert not waiting.done()
    await subscriber.aclose()
    await asyncio.wait_for(waiting, timeout=1.0)
    await bus.shutdown()

  asyncio.run(_run())


def test_a_stalled_watcher_costs_the_runs_one_deadline_not_one_per_event() -> None:
  """DEFECT-DESK baf4e9fa: the deadline ends a backlog episode. After one
  bounded wait on a subscriber that stopped reading, later publishes neither
  wait nor stall the producing run (autonomous event recording awaits publish
  inline); the bus drops oldest with the sentinel instead. Once the consumer
  drains below the mark, backpressure applies again."""
  async def _run() -> None:
    bus = UserEventBus(subscriber_queue_max=4, backpressure_wait_seconds=0.05)
    subscriber = _subscribe(bus, "alice")  # global watcher
    first_task = asyncio.create_task(subscriber.__anext__())
    await asyncio.sleep(0)
    await bus.publish("alice", "run-1", {"type": "message", "seq": 0})
    assert await asyncio.wait_for(first_task, timeout=0.5) == {"type": "message", "seq": 0}

    # The watcher stops reading: one publish pays the deadline (the episode).
    await bus.publish("alice", "run-1", {"type": "message", "seq": 1})
    await bus.publish("alice", "run-1", {"type": "message", "seq": 2})

    # Every later publish completes without waiting on the stalled watcher.
    for seq in range(3, 23):
      publishing = asyncio.create_task(bus.publish("alice", "run-1", {"type": "message", "seq": seq}))
      for _ in range(5):
        await asyncio.sleep(0)
      assert publishing.done(), f"publish {seq} waited on a stalled watcher"

    # The watcher resumes: the sentinel names the loss, then the newest events.
    sentinel = await _next_event(subscriber)
    assert sentinel["type"] == "events_dropped"
    tail = [await _next_event(subscriber) for _ in range(4)]
    assert [event["seq"] for event in tail] == [19, 20, 21, 22]

    # Drained below the mark: the next backlog episode waits again.
    await bus.publish("alice", "run-1", {"type": "message", "seq": 23})
    waiting = asyncio.create_task(bus.publish("alice", "run-1", {"type": "message", "seq": 24}))
    for _ in range(5):
      await asyncio.sleep(0)
    assert not waiting.done()
    assert await _next_event(subscriber) == {"type": "message", "seq": 23}
    await asyncio.wait_for(waiting, timeout=1.0)

    await subscriber.aclose()
    await bus.shutdown()

  asyncio.run(_run())


def test_a_watcher_that_pauses_once_and_resumes_gets_backpressure_back() -> None:
  """Audit 37a74abb489b P2: after one deadline the episode was spent until the
  queue fell below the mark, which a resumed watcher slower than its producers
  never reaches, so it stayed on drop-oldest. Reading half a bound re-arms the
  wait while producers are still publishing."""
  async def _run() -> None:
    bus = UserEventBus(subscriber_queue_max=8, backpressure_wait_seconds=0.05)
    subscriber = _subscribe(bus, "alice")
    first = asyncio.create_task(subscriber.__anext__())
    await asyncio.sleep(0)
    await bus.publish("alice", "run-1", {"type": "message", "seq": 0})
    await asyncio.wait_for(first, timeout=0.5)

    # The watcher pauses past the deadline while the run keeps publishing.
    for seq in range(1, 9):
      await bus.publish("alice", "run-1", {"type": "message", "seq": seq})

    # It resumes reading; the run is still publishing between its reads.
    seq = 9
    # One publish per read: every publish evicts, so every read is the
    # events_dropped sentinel. Those deliveries prove the watcher is live.
    for _ in range(4):  # half the bound of 8
      await bus.publish("alice", "run-1", {"type": "message", "seq": seq})  # spent: no wait
      seq += 1
      await _next_event(subscriber)

    # Backpressure is back: the next publish past the mark waits for the reader.
    waiting = asyncio.create_task(bus.publish("alice", "run-1", {"type": "message", "seq": seq}))
    for _ in range(5):
      await asyncio.sleep(0)
    assert not waiting.done()
    for _ in range(5):  # reading below the mark (8 -> 3) releases the producer
      await _next_event(subscriber)
    await asyncio.wait_for(waiting, timeout=1.0)
    await subscriber.aclose()
    await bus.shutdown()

  asyncio.run(_run())


def test_runs_ending_do_not_shrink_a_global_watchers_allotment() -> None:
  """Audit 19c1e0daa6b0 P2: the global bound counted live runs only, so when
  runs ended the bound fell below the queued backlog and the next publish from
  a surviving run evicted events the live watcher was entitled to."""
  async def _run() -> None:
    bus = UserEventBus(subscriber_queue_max=2, backpressure_wait_seconds=0)
    subscriber = _subscribe(bus, "alice")
    first = asyncio.create_task(subscriber.__anext__())
    await asyncio.sleep(0)
    await bus.publish("alice", "run-0", {"type": "message", "run": 0, "i": -1})
    await asyncio.wait_for(first, timeout=0.5)

    for r in range(5):  # five live runs: bound 10
      await bus.publish("alice", f"run-{r}", {"type": "message", "run": r, "i": 0})
    for r in range(5):
      await bus.publish("alice", f"run-{r}", {"type": "message", "run": r, "i": 1})
    # 10 queued; drain two so the backlog (8) sits below the 5-run bound.
    await _next_event(subscriber)
    await _next_event(subscriber)
    for r in range(1, 5):
      await bus.cleanup_run("alice", f"run-{r}")

    await bus.publish("alice", "run-0", {"type": "message", "run": 0, "i": 2})

    received = [await _next_event(subscriber) for _ in range(9)]
    assert all(event["type"] == "message" for event in received)
    assert received[-1] == {"type": "message", "run": 0, "i": 2}
    await subscriber.aclose()
    await bus.shutdown()

  asyncio.run(_run())


def test_user_event_bus_replays_buffer_before_live_tail() -> None:
  async def _run() -> None:
    bus = UserEventBus(replay_buffer_max=2)
    await bus.publish("alice", "run-1", {"type": "message", "seq": 1})
    await bus.publish("alice", "run-1", {"type": "message", "seq": 2})
    await bus.publish("alice", "run-1", {"type": "message", "seq": 3})

    subscriber = _subscribe(bus, "alice", control_run_id="run-1")
    assert await _next_event(subscriber) == {"type": "message", "seq": 2}
    assert await _next_event(subscriber) == {"type": "message", "seq": 3}

    live_task = asyncio.create_task(subscriber.__anext__())
    await asyncio.sleep(0)
    await bus.publish("alice", "run-1", {"type": "message", "seq": 4})
    assert await asyncio.wait_for(live_task, timeout=0.5) == {"type": "message", "seq": 4}

    await subscriber.aclose()
    await bus.shutdown()

  asyncio.run(_run())


def test_user_event_bus_seeds_replay_buffer_from_durable_events() -> None:
  async def _run() -> None:
    bus = UserEventBus(replay_buffer_max=2)
    seeded = await bus.seed_replay_buffer(
      "alice",
      "run-1",
      [
        {"type": "message", "value": "one"},
        {"type": "message", "value": "two"},
        {"type": "message", "value": "three"},
      ],
    )
    assert seeded == 2

    subscriber = await bus.subscribe_entries("alice", control_run_id="run-1", after_seq=0)
    truncated = await asyncio.wait_for(subscriber.__anext__(), timeout=0.5)
    assert truncated.seq is None
    assert truncated.event == {
      "type": "replay_truncated",
      "run_id": "run-1",
      "control_run_id": "run-1",
      "dropped_before_seq": 2,
    }
    replayed = [await asyncio.wait_for(subscriber.__anext__(), timeout=0.5) for _ in range(2)]
    assert [(entry.seq, entry.event["value"]) for entry in replayed] == [(2, "two"), (3, "three")]

    live_task = asyncio.create_task(subscriber.__anext__())
    await asyncio.sleep(0)
    await bus.publish("alice", "run-1", {"type": "message", "value": "four"})
    live = await asyncio.wait_for(live_task, timeout=0.5)
    assert live.seq == 4
    assert live.event == {"type": "message", "value": "four"}

    assert await bus.seed_replay_buffer("alice", "run-1", [{"type": "message", "value": "duplicate"}]) == 0
    await subscriber.aclose()
    await bus.shutdown()

  asyncio.run(_run())


def test_terminal_publish_is_atomic_with_durable_replay_seed() -> None:
  async def _run() -> None:
    bus = UserEventBus()
    terminal = {
      "type": "run_state_changed",
      "state": "completed",
    }
    assert await bus.seed_replay_buffer(
      "alice",
      "batch_1",
      [terminal],
      terminated=True,
    ) == 1
    assert await bus.publish_terminal_if_absent(
      "alice",
      "batch_1",
      terminal,
    ) is False

    subscriber = await bus.subscribe_entries(
      "alice",
      control_run_id="batch_1",
      after_seq=0,
    )
    replayed = await asyncio.wait_for(
      subscriber.__anext__(),
      timeout=0.5,
    )
    assert replayed.seq == 1
    assert replayed.event == terminal
    with pytest.raises(StopAsyncIteration):
      await subscriber.__anext__()
    await subscriber.aclose()
    await bus.shutdown()

  asyncio.run(_run())


def test_terminal_publish_delivers_once_and_schedules_cleanup() -> None:
  async def _run() -> None:
    bus = UserEventBus()
    terminal = {
      "type": "run_state_changed",
      "state": "failed",
    }
    subscriber = await bus.subscribe_entries(
      "alice",
      control_run_id="batch_2",
    )

    assert await bus.publish_terminal_if_absent(
      "alice",
      "batch_2",
      terminal,
    ) is True
    assert await bus.publish_terminal_if_absent(
      "alice",
      "batch_2",
      terminal,
    ) is False
    delivered = await asyncio.wait_for(
      subscriber.__anext__(),
      timeout=0.5,
    )
    assert delivered.seq == 1
    assert delivered.event == terminal
    with pytest.raises(StopAsyncIteration):
      await subscriber.__anext__()
    await subscriber.aclose()
    await bus.shutdown()

  asyncio.run(_run())


def test_terminal_publish_appends_after_live_prefix_and_stops_follower() -> None:
  async def _run() -> None:
    bus = UserEventBus()
    subscriber = await bus.subscribe_entries(
      "alice",
      control_run_id="batch_3",
    )
    terminal = {
      "type": "run_state_changed",
      "state": "completed",
    }

    await bus.publish(
      "alice",
      "batch_3",
      {"type": "run_state_changed", "state": "running"},
    )
    prefix = await asyncio.wait_for(
      subscriber.__anext__(),
      timeout=0.5,
    )
    assert prefix.seq == 1
    assert prefix.event["state"] == "running"
    assert await bus.publish_terminal_if_absent(
      "alice",
      "batch_3",
      terminal,
    ) is True
    assert await bus.publish_terminal_if_absent(
      "alice",
      "batch_3",
      terminal,
    ) is False
    delivered = await asyncio.wait_for(
      subscriber.__anext__(),
      timeout=0.5,
    )
    assert delivered.seq == 2
    assert delivered.event == terminal
    with pytest.raises(StopAsyncIteration):
      await subscriber.__anext__()
    await subscriber.aclose()
    await bus.shutdown()

  asyncio.run(_run())


def test_user_event_bus_seed_existing_buffer_can_mark_terminal_for_cleanup() -> None:
  async def _run() -> None:
    bus = UserEventBus()
    bus._cleanup_delay_seconds = 0.01
    assert await bus.seed_replay_buffer("alice", "run-1", [{"type": "message", "value": "one"}]) == 1
    assert await bus.seed_replay_buffer("alice", "run-1", [], terminated=True) == 0
    await asyncio.sleep(0.05)

    subscriber = _subscribe(bus, "alice", control_run_id="run-1")
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


def test_user_event_bus_fast_run_race_replays_after_termination_before_cleanup() -> None:
  async def _run() -> None:
    bus = UserEventBus()
    bus._cleanup_delay_seconds = 0.5
    await bus.publish("alice", "bg_1", {"type": "run_state_changed", "state": "running"})
    await bus.publish("alice", "bg_1", {"type": "run_state_changed", "state": "complete"})
    await bus.cleanup_run("alice", "bg_1")

    subscriber = _subscribe(bus, "alice", control_run_id="bg_1")
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

    subscriber = _subscribe(bus, "alice", control_run_id="run-1")
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
    subscriber = _subscribe(bus, "alice", control_run_id="run-1")
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
