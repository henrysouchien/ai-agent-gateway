from __future__ import annotations

import asyncio
from typing import Any, AsyncIterator, Callable

from .event_adapter import adapt_event
from .event_log import EventLog
from .events import DEFAULT_SCHEMA_VERSION, SUPPORTED_SCHEMA_VERSIONS
from .product_config import gateway_product_id
from .server_artifact_helpers import _json_dumps
from .session import GatewaySession, SessionStream, StreamSubscriber


def event_for_wire(
  entry: Any,
  event_log: EventLog,
  *,
  product_id_resolver: Callable[[], str | None] = gateway_product_id,
) -> dict[str, Any]:
  event = dict(entry.event)
  pid = product_id_resolver()
  if pid is not None:
    event["product_id"] = pid
  if event.get("type") in {"tool_call_start", "tool_call_complete"}:
    tool_name = event.get("tool_name")
    execution_location_resolver = getattr(event_log, "_gateway_execution_location", None)
    if isinstance(tool_name, str) and execution_location_resolver is not None:
      execution_location = execution_location_resolver(tool_name)
      if execution_location is not None:
        event["execution_location"] = execution_location
  return event


def resolve_schema_version(
  schema_version: int | None,
  *,
  default_schema_version: int = DEFAULT_SCHEMA_VERSION,
  supported_schema_versions: set[int] | frozenset[int] = SUPPORTED_SCHEMA_VERSIONS,
  http_exception_cls: Any,
) -> int:
  resolved = default_schema_version if schema_version is None else int(schema_version)
  if resolved not in supported_schema_versions:
    supported = ", ".join(str(version) for version in sorted(supported_schema_versions))
    raise http_exception_cls(
      status_code=400,
      detail=f"Unsupported schema_version {resolved}; supported: [{supported}]",
    )
  return resolved


def stream_envelope(*, entry: Any, session_id: str, schema_version: int, event: dict[str, Any]) -> dict[str, Any]:
  return {
    "seq": entry.seq,
    "session_id": session_id,
    "schema_version": schema_version,
    "event": event,
  }


def disconnect_stream_subscriber_for_backpressure(
  subscriber: StreamSubscriber,
  *,
  done_marker: object,
) -> None:
  subscriber.disconnect_reason = "backpressure"
  while True:
    try:
      subscriber.queue.get_nowait()
    except asyncio.QueueEmpty:
      break
  try:
    subscriber.queue.put_nowait(done_marker)
  except asyncio.QueueFull:
    pass


async def pump_stream_subscriber(
  active_turn: SessionStream,
  subscriber: StreamSubscriber,
  after_seq: int,
  *,
  done_marker: object,
  disconnect_stream_subscriber_for_backpressure: Callable[[StreamSubscriber], None],
) -> None:
  try:
    async for entry in active_turn.event_log.iter_from(after_seq):
      try:
        subscriber.queue.put_nowait(entry)
      except asyncio.QueueFull:
        disconnect_stream_subscriber_for_backpressure(subscriber)
        return
  except asyncio.CancelledError:
    raise
  else:
    try:
      subscriber.queue.put_nowait(done_marker)
    except asyncio.QueueFull:
      disconnect_stream_subscriber_for_backpressure(subscriber)


def register_stream_subscriber(
  active_turn: SessionStream,
  *,
  after_seq: int,
  client_label: str | None,
  queue_max: int,
  queue_factory: Callable[..., Any],
  task_factory: Callable[[Any], Any],
  uuid_hex_factory: Callable[[], str],
  time_fn: Callable[[], float],
  default_pump_stream_subscriber: Callable[[SessionStream, StreamSubscriber, int], Any],
  pump_stream_subscriber: Callable[[SessionStream, StreamSubscriber, int], Any] | None = None,
) -> StreamSubscriber:
  subscriber = StreamSubscriber(
    subscriber_id=f"sub:{uuid_hex_factory()}",
    connected_at=time_fn(),
    last_sent_seq=max(int(after_seq), 0),
    queue=queue_factory(maxsize=queue_max),
    client_label=client_label,
  )
  active_turn.subscribers[subscriber.subscriber_id] = subscriber
  pump = default_pump_stream_subscriber if pump_stream_subscriber is None else pump_stream_subscriber
  subscriber.pump_task = task_factory(pump(active_turn, subscriber, subscriber.last_sent_seq))
  return subscriber


async def cleanup_stream_subscriber(
  active_turn: SessionStream,
  subscriber_id: str,
  *,
  gather_fn: Callable[..., Any],
) -> None:
  subscriber = active_turn.subscribers.pop(subscriber_id, None)
  if subscriber is None:
    return
  pump_task = subscriber.pump_task
  if pump_task is not None and not pump_task.done():
    pump_task.cancel()
    await gather_fn(pump_task, return_exceptions=True)


async def stream_subscriber_sse(
  *,
  session: GatewaySession,
  active_turn: SessionStream,
  subscriber: StreamSubscriber,
  transcript_dir: Any,
  channel: str | None,
  write_transcript: bool,
  log: Any,
  keepalive_seconds: float,
  done_marker: object,
  cleanup_stream_subscriber: Callable[[SessionStream, str], Any],
  wait_for_fn: Callable[..., Any],
  timeout_error: type[BaseException],
  event_for_wire_fn: Callable[[Any, EventLog], dict[str, Any]],
  write_transcript_fn: Callable[..., None],
  adapt_event_fn: Callable[[dict[str, Any], int], dict[str, Any] | None] = adapt_event,
  stream_envelope_fn: Callable[..., dict[str, Any]] = stream_envelope,
  json_dumps_fn: Callable[[Any], str] = _json_dumps,
  default_schema_version: int = DEFAULT_SCHEMA_VERSION,
) -> AsyncIterator[bytes]:
  event_log = active_turn.event_log
  try:
    while True:
      try:
        item = await wait_for_fn(
          subscriber.queue.get(),
          timeout=keepalive_seconds,
        )
      except timeout_error:
        pump_task = subscriber.pump_task
        if pump_task is not None and pump_task.done() and subscriber.disconnect_reason:
          return
        yield b":keepalive\n\n"
        continue

      if item is done_marker:
        return

      entry = item
      subscriber.last_sent_seq = int(entry.seq)
      event = event_for_wire_fn(entry, event_log)
      entry_seq = int(entry.seq)
      if write_transcript and entry_seq not in active_turn.transcript_written_seqs:
        active_turn.transcript_written_seqs.add(entry_seq)
        write_transcript_fn(
          transcript_dir=transcript_dir,
          session_id=session.session_id,
          entry=event,
          user_id=session.user_id,
          channel=channel,
        )

      try:
        adapted_event = adapt_event_fn(event, session.schema_version)
      except ValueError as adapter_exc:
        log.error(
          "SSE adapter failed for session=%s schema_version=%s event_type=%s: %s",
          session.session_id,
          session.schema_version,
          event.get("type"),
          adapter_exc,
          exc_info=True,
        )
        error_event = {"type": "stream_error", "error": str(adapter_exc)}
        adapted_event = adapt_event_fn(error_event, default_schema_version)
      if adapted_event is None:
        continue

      envelope = stream_envelope_fn(
        entry=entry,
        session_id=session.session_id,
        schema_version=session.schema_version,
        event=adapted_event,
      )
      try:
        yield f"data: {json_dumps_fn(envelope)}\n\n".encode("utf-8")
      except Exception as ser_exc:
        log.error(
          "SSE serialization failed for event type=%s: %s",
          event.get("type"),
          ser_exc,
          exc_info=True,
        )
        try:
          error_event = {"type": "stream_error", "error": f"SSE serialization failed: {ser_exc}"}
          error_envelope = {
            "seq": subscriber.last_sent_seq,
            "session_id": session.session_id,
            "schema_version": session.schema_version,
            "event": adapt_event_fn(error_event, session.schema_version) or error_event,
          }
          yield f"data: {json_dumps_fn(error_envelope)}\n\n".encode("utf-8")
        except Exception:
          pass
        return
  finally:
    await cleanup_stream_subscriber(active_turn, subscriber.subscriber_id)


__all__ = [
  "cleanup_stream_subscriber",
  "disconnect_stream_subscriber_for_backpressure",
  "event_for_wire",
  "pump_stream_subscriber",
  "register_stream_subscriber",
  "resolve_schema_version",
  "stream_envelope",
  "stream_subscriber_sse",
]
