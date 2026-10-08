"""Bounded host/client frames share one executing-cell budget, not one token use."""
import asyncio
import sys
from pathlib import Path

import pytest

PKG_DIR = Path(__file__).resolve().parents[1]
if str(PKG_DIR) not in sys.path:
  sys.path.insert(0, str(PKG_DIR))

from agent_gateway.code_execution import _frame_codec as codec
from agent_gateway.code_execution._bridge_client import HankBridge, HankBridgeError
from agent_gateway.code_execution._frame_reader import FrameReader


def test_ruled_byte_limits_and_utf8_wire_size():
  assert codec.MAX_FRAME_BYTES == 2 * 1024 * 1024
  assert codec.MAX_CELL_BYTES == 64 * 1024 * 1024
  frame = {"text": "é" * 30}
  wire = codec.encode_frame(frame)
  budget = codec.CellBudget()
  assert codec.decode_frame(wire, budget) == frame
  assert budget.used_bytes == len(wire)
  with pytest.raises(codec.ProtocolError, match="2 MiB"):
    codec.encode_frame({"text": "é" * codec.MAX_FRAME_BYTES})


def test_decode_rejects_before_json_allocation(monkeypatch):
  called = []
  monkeypatch.setattr(codec.json, "loads", lambda line: called.append(line))
  with pytest.raises(codec.ProtocolError, match="2 MiB"):
    codec.decode_frame(b"x" * (codec.MAX_FRAME_BYTES + 1))
  assert called == []


def test_exact_frame_boundary_and_poisoned_aggregate():
  wire = codec.encode_frame({"text": "x" * (codec.MAX_FRAME_BYTES - len(b'{"text":""}\n'))})
  assert len(wire) == codec.MAX_FRAME_BYTES
  budget = codec.CellBudget()
  for _ in range(32):
    codec.decode_frame(wire, budget)
  assert budget.used_bytes == codec.MAX_CELL_BYTES
  with pytest.raises(codec.ProtocolError, match="64 MiB"):
    codec.decode_frame(b"{}\n", budget)
  with pytest.raises(codec.ProtocolError, match="64 MiB"):
    codec.encode_frame({}, budget)
  assert codec.decode_frame(codec.encode_frame({"error": "protocol_error"})) == {"error": "protocol_error"}


def test_direct_reader_bounds_and_accounted_transport_do_not_double_charge():
  async def run():
    reader = FrameReader()
    cell = reader.open_cell("outer", abort_event=None, active=True)
    try:
      frame = {"type": "hello", "call_id": "h1", "cell_token": cell.token, "protocol": 1}
      wire = codec.encode_frame(frame)
      codec.decode_frame(wire, cell.budget)
      reply_wire = await cell.handle_frame(wire, accounted=True)
      assert codec.decode_frame(reply_wire)["ok"] is True
      assert cell.budget.used_bytes == len(wire) + len(reply_wire)
      oversized = await cell.answer_frame(b"x" * (codec.MAX_FRAME_BYTES + 1), accounted=True)
      assert oversized["error"]["code"] == "protocol_error"
      # Even an accounted call cannot bypass the byte bound. A fresh cell has
      # its own aggregate and token; the interpreter need not be destroyed.
      fresh = reader.open_cell("later", abort_event=None, active=True)
      try:
        reply = await fresh.answer_frame(codec.encode_frame({"type": "hello", "cell_token": fresh.token}))
        assert reply["ok"] is True
      finally:
        fresh.close()
    finally:
      cell.close()
  asyncio.run(run())


def test_client_receives_aggregate_protocol_error_not_silent_close():
  async def run():
    reader = FrameReader()
    cell = reader.open_cell("outer", abort_event=None, active=True)
    loop = asyncio.get_running_loop()

    class Transport:
      def exchange(self, frame):
        request = codec.encode_frame(frame)
        reply = asyncio.run_coroutine_threadsafe(cell.handle_frame(request), loop).result()
        return codec.decode_frame(reply)

    client = HankBridge(Transport(), cell.token)
    try:
      # Fill a real (small test) aggregate with a stdout frame, sharing the
      # very same budget the reader charges for request and reply frames.
      cell.budget = codec.CellBudget(aggregate_limit=1024)
      await asyncio.to_thread(client._open)
      remaining = cell.budget.aggregate_limit - cell.budget.used_bytes
      overhead = len(codec.encode_frame({"type": "stdout", "text": ""}))
      output = codec.encode_frame({"type": "stdout", "text": "x" * (remaining - overhead)})
      codec.decode_frame(output, cell.budget)
      with pytest.raises(HankBridgeError) as failure:
        await asyncio.to_thread(client.search, "filings")
      assert failure.value.code == "protocol_error"
      assert cell.budget.error is not None
      with pytest.raises(HankBridgeError) as again:
        await asyncio.to_thread(client.search, "filings")
      assert again.value.code == "protocol_error"
    finally:
      cell.close()
  asyncio.run(run())


def test_pipe_client_bounds_reply_before_decode():
  import io
  from agent_gateway.code_execution._bridge_client import _PipeTransport

  reader = io.BytesIO(b"x" * (codec.MAX_FRAME_BYTES + 1))
  writer = io.BytesIO()
  client = HankBridge(_PipeTransport(reader, writer), "cell")
  with pytest.raises(HankBridgeError) as failure:
    client.search("anything")
  assert failure.value.code == "protocol_error"
  assert len(writer.getvalue()) < codec.MAX_FRAME_BYTES


def test_malformed_frame_error_counts_reply_and_allows_next_valid_request():
  async def run():
    reader = FrameReader()
    cell = reader.open_cell("outer", abort_event=None, active=True)
    try:
      malformed = b"not-json\n"
      error = await cell.handle_frame(malformed)
      assert codec.decode_frame(error)["error"]["code"] == "protocol_error"
      assert cell.budget.used_bytes == len(malformed) + len(error)
      request = codec.encode_frame({"type": "hello", "call_id": "valid", "cell_token": cell.token})
      reply = await cell.handle_frame(request)
      assert codec.decode_frame(reply)["ok"]
      assert cell.budget.used_bytes == len(malformed) + len(error) + len(request) + len(reply)
    finally:
      cell.close()
  asyncio.run(run())


def test_terminal_error_cannot_echo_oversized_correlation_fields():
  async def run():
    reader = FrameReader()
    # A long registered parent id appears in the hello reply, not its request.
    # Thus a valid request at the exact cap necessarily has an oversized reply.
    cell = reader.open_cell("outer-parent-" * 40, abort_event=None, active=True)
    try:
      frame = {"type": "hello", "call_id": "", "cell_token": cell.token}
      size = len(codec.encode_frame(frame))
      frame["call_id"] = "x" * (codec.MAX_FRAME_BYTES - size)
      request = codec.encode_frame(frame)
      assert len(request) == codec.MAX_FRAME_BYTES
      wire = await cell.handle_frame(request)
      assert len(wire) < codec.MAX_FRAME_BYTES
      assert codec.decode_frame(wire)["error"]["code"] == "protocol_error"
      assert cell.budget.error is not None
    finally:
      cell.close()
  asyncio.run(run())
