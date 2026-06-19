from __future__ import annotations

from collections.abc import AsyncIterator, Callable, Mapping
from dataclasses import dataclass
from typing import Any

from .session import StreamSubscriber


@dataclass(frozen=True)
class StreamingDeps:
  chat_helpers: Any
  done_marker: object
  queue_max: int
  keepalive_seconds: float
  disconnect_stream_subscriber_for_backpressure: Callable[[StreamSubscriber], None]
  pump_stream_subscriber: Callable[..., Any]
  cleanup_stream_subscriber: Callable[..., Any]


def _deps(namespace: Mapping[str, Any]) -> StreamingDeps:
  return StreamingDeps(
    chat_helpers=namespace["_server_chat_helpers"],
    done_marker=namespace["_STREAM_SUBSCRIBER_DONE"],
    queue_max=namespace["_STREAM_SUBSCRIBER_QUEUE_MAX"],
    keepalive_seconds=namespace["_STREAM_SUBSCRIBER_KEEPALIVE_SECONDS"],
    disconnect_stream_subscriber_for_backpressure=namespace[
      "_disconnect_stream_subscriber_for_backpressure"
    ],
    pump_stream_subscriber=namespace["_pump_stream_subscriber"],
    cleanup_stream_subscriber=namespace["_cleanup_stream_subscriber"],
  )


def bind_streaming_helpers(namespace: Callable[[], Mapping[str, Any]]) -> tuple[
  Callable[[StreamSubscriber], None],
  Callable[[Any, StreamSubscriber, int], Any],
  Callable[..., StreamSubscriber],
  Callable[[Any, str], Any],
  Callable[..., AsyncIterator[Any]],
]:
  def disconnect(subscriber: StreamSubscriber) -> None:
    return disconnect_stream_subscriber_for_backpressure(
      subscriber,
      deps=_deps(namespace()),
    )

  async def pump(active_turn: Any, subscriber: StreamSubscriber, after_seq: int) -> None:
    await pump_stream_subscriber(
      active_turn,
      subscriber,
      after_seq,
      deps=_deps(namespace()),
    )

  def register(active_turn: Any, *, after_seq: int, client_label: str | None) -> StreamSubscriber:
    return register_stream_subscriber(
      active_turn,
      after_seq=after_seq,
      client_label=client_label,
      deps=_deps(namespace()),
    )

  async def cleanup(active_turn: Any, subscriber_id: str) -> None:
    await cleanup_stream_subscriber(
      active_turn,
      subscriber_id,
      deps=_deps(namespace()),
    )

  async def sse(**kwargs: Any) -> AsyncIterator[Any]:
    async for chunk in stream_subscriber_sse(
      **kwargs,
      deps=_deps(namespace()),
    ):
      yield chunk

  module_name = str(namespace().get("__name__", __name__))
  for func, name in (
    (disconnect, "_disconnect_stream_subscriber_for_backpressure"),
    (pump, "_pump_stream_subscriber"),
    (register, "_register_stream_subscriber"),
    (cleanup, "_cleanup_stream_subscriber"),
    (sse, "_stream_subscriber_sse"),
  ):
    func.__name__ = name
    func.__qualname__ = name
    func.__module__ = module_name

  return disconnect, pump, register, cleanup, sse


def disconnect_stream_subscriber_for_backpressure(
  subscriber: StreamSubscriber,
  *,
  deps: StreamingDeps,
) -> None:
  return deps.chat_helpers._disconnect_stream_subscriber_for_backpressure(
    subscriber,
    done_marker=deps.done_marker,
  )


async def pump_stream_subscriber(
  active_turn: Any,
  subscriber: StreamSubscriber,
  after_seq: int,
  *,
  deps: StreamingDeps,
) -> None:
  await deps.chat_helpers._pump_stream_subscriber(
    active_turn,
    subscriber,
    after_seq,
    done_marker=deps.done_marker,
    disconnect_stream_subscriber_for_backpressure=deps.disconnect_stream_subscriber_for_backpressure,
  )


def register_stream_subscriber(
  active_turn: Any,
  *,
  after_seq: int,
  client_label: str | None,
  deps: StreamingDeps,
) -> StreamSubscriber:
  return deps.chat_helpers._register_stream_subscriber(
    active_turn,
    after_seq=after_seq,
    client_label=client_label,
    queue_max=deps.queue_max,
    pump_stream_subscriber=deps.pump_stream_subscriber,
  )


async def cleanup_stream_subscriber(
  active_turn: Any,
  subscriber_id: str,
  *,
  deps: StreamingDeps,
) -> None:
  await deps.chat_helpers._cleanup_stream_subscriber(active_turn, subscriber_id)


async def stream_subscriber_sse(
  **kwargs: Any,
) -> AsyncIterator[Any]:
  deps = kwargs.pop("deps")
  async for chunk in deps.chat_helpers._stream_subscriber_sse(
    **kwargs,
    keepalive_seconds=deps.keepalive_seconds,
    done_marker=deps.done_marker,
    cleanup_stream_subscriber=deps.cleanup_stream_subscriber,
  ):
    yield chunk
