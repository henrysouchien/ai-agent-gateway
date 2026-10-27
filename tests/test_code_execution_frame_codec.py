"""Frames share one executing-cell aggregate budget; one big frame never poisons it."""
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

_MIB = 1024 * 1024


def _catalog_frame(cell, call_id: str) -> dict:
  # A reader with no bound dispatcher answers a catalog request with a small,
  # deterministic correlated error reply: a real request/reply wire round trip.
  return {
    "type": "catalog", "request": "search", "query": "", "call_id": call_id,
    "cell_token": cell.token, "parent_call_id": cell.outer_call_id,
  }


def test_ruled_byte_limits_and_utf8_wire_size():
  assert codec.MAX_FRAME_BYTES == 2 * _MIB
  assert codec.MAX_CELL_BYTES == 64 * _MIB
  frame = {"text": "é" * 30}
  wire = codec.encode_frame(frame)
  budget = codec.CellBudget()
  assert codec.decode_frame(wire, budget) == frame
  assert budget.used_bytes == len(wire)
  # The 2 MiB display bound is an explicit per-call limit counted in UTF-8 bytes.
  with pytest.raises(codec.ProtocolError, match="2 MiB"):
    codec.encode_frame({"text": "é" * codec.MAX_FRAME_BYTES}, limit=codec.MAX_FRAME_BYTES)


def test_decode_rejects_before_json_allocation(monkeypatch):
  called = []
  monkeypatch.setattr(codec.json, "loads", lambda line: called.append(line))
  with pytest.raises(codec.ProtocolError, match="2 MiB"):
    codec.decode_frame(b"x" * (codec.MAX_FRAME_BYTES + 1), limit=codec.MAX_FRAME_BYTES)
  with pytest.raises(codec.ProtocolError):
    codec.decode_frame(b"x" * 9, limit=8)
  assert called == []


def test_single_frame_above_two_mib_roundtrips_under_the_cell_limit():
  frame = {"type": "tool_result", "ok": True, "result": {"text": "é" * (3 * _MIB)}}
  wire = codec.encode_frame(frame)
  assert len(wire) > 6 * _MIB > codec.MAX_FRAME_BYTES
  budget = codec.CellBudget()
  assert codec.decode_frame(wire, budget) == frame
  assert budget.used_bytes == len(wire)
  assert budget.error is None
  reply = codec.encode_frame(frame, budget)
  assert reply == wire
  assert budget.used_bytes == 2 * len(wire)
  assert budget.error is None


def test_frame_over_its_limit_raises_without_poisoning_the_budget():
  frame = {"text": "x" * (3 * _MIB)}
  wire = codec.encode_frame(frame)
  budget = codec.CellBudget()
  with pytest.raises(codec.ProtocolError, match="2 MiB"):
    codec.decode_frame(wire, budget, limit=codec.MAX_FRAME_BYTES)
  with pytest.raises(codec.ProtocolError, match="2 MiB"):
    codec.encode_frame(frame, budget, limit=codec.MAX_FRAME_BYTES)
  assert budget.error is None
  assert budget.used_bytes == 0
  budget.charge(len(wire))  # a single large charge below the aggregate is not an error
  assert budget.error is None
  assert budget.used_bytes == len(wire)
  # The same budget still carries the next, valid frames in both directions.
  small = codec.encode_frame({"ok": True}, budget)
  assert codec.decode_frame(small, budget) == {"ok": True}
  assert budget.used_bytes == len(wire) + 2 * len(small)
  assert budget.error is None


def test_exact_frame_boundary_and_poisoned_aggregate():
  text = "x" * (codec.MAX_FRAME_BYTES - len(b'{"text":""}\n'))
  wire = codec.encode_frame({"text": text}, limit=codec.MAX_FRAME_BYTES)
  assert len(wire) == codec.MAX_FRAME_BYTES
  with pytest.raises(codec.ProtocolError, match="2 MiB"):
    codec.encode_frame({"text": text + "x"}, limit=codec.MAX_FRAME_BYTES)
  budget = codec.CellBudget()
  for _ in range(32):
    codec.decode_frame(wire, budget)
  assert budget.used_bytes == codec.MAX_CELL_BYTES
  assert budget.error is None
  with pytest.raises(codec.ProtocolError, match="aggregate"):
    codec.decode_frame(b"{}\n", budget)
  assert budget.error is not None
  with pytest.raises(codec.ProtocolError, match="aggregate"):
    codec.encode_frame({}, budget)
  assert codec.decode_frame(codec.encode_frame({"error": "protocol_error"})) == {"error": "protocol_error"}


def test_cumulative_frames_past_64_mib_set_the_budget_error():
  wire = codec.encode_frame({"text": "x" * (3 * _MIB)})
  budget = codec.CellBudget()
  fitting = codec.MAX_CELL_BYTES // len(wire)
  assert 0 < fitting < 32
  for _ in range(fitting):
    codec.decode_frame(wire, budget)
  assert budget.error is None
  assert budget.used_bytes == fitting * len(wire)
  with pytest.raises(codec.ProtocolError, match="aggregate"):
    codec.decode_frame(wire, budget)
  assert "64 MiB" in (budget.error or "")
  # The breaching frame is not charged, and the poison is permanent for the cell.
  assert budget.used_bytes == fitting * len(wire)
  with pytest.raises(codec.ProtocolError, match="aggregate"):
    codec.decode_frame(b"{}\n", budget)


def test_reader_oversized_request_is_a_correlated_error_and_the_cell_continues():
  async def run():
    reader = FrameReader()
    cell = reader.open_cell("outer", abort_event=None, active=True)
    try:
      wire = codec.encode_frame(_catalog_frame(cell, "h1"))
      reply_wire = await cell.handle_frame(wire)
      reply = codec.decode_frame(reply_wire)
      assert reply["call_id"] == "h1"
      assert reply["error"]["code"] == "bridge_unavailable"
      assert cell.budget.used_bytes == len(wire) + len(reply_wire)
      before = cell.budget.used_bytes
      error_wire = await cell.handle_frame(b"x" * (codec.MAX_CELL_BYTES + 1))
      assert codec.decode_frame(error_wire)["error"]["code"] == "protocol_error"
      # One over-limit request is that request's error: the budget is not poisoned.
      assert cell.budget.error is None
      assert cell.budget.used_bytes == before + len(error_wire)
      again = codec.decode_frame(await cell.handle_frame(codec.encode_frame(_catalog_frame(cell, "h2"))))
      assert again["call_id"] == "h2"
      assert again["error"]["code"] == "bridge_unavailable"
      # A fresh cell has its own aggregate and token; the interpreter survives.
      fresh = reader.open_cell("later", abort_event=None, active=True)
      try:
        reply = codec.decode_frame(await fresh.handle_frame(codec.encode_frame(_catalog_frame(fresh, "h3"))))
        assert reply["call_id"] == "h3"
        assert reply["error"]["code"] == "bridge_unavailable"
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
      cell.budget = codec.CellBudget(aggregate_limit=4096)
      overhead = len(codec.encode_frame({"type": "stdout", "text": ""}))
      output = codec.encode_frame({"type": "stdout", "text": "x" * (cell.budget.aggregate_limit - overhead)})
      codec.decode_frame(output, cell.budget)
      assert cell.budget.error is None
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


def test_malformed_frame_error_counts_reply_and_allows_next_valid_request():
  async def run():
    reader = FrameReader()
    cell = reader.open_cell("outer", abort_event=None, active=True)
    try:
      malformed = b"not-json\n"
      error = await cell.handle_frame(malformed)
      assert codec.decode_frame(error)["error"]["code"] == "protocol_error"
      assert cell.budget.used_bytes == len(malformed) + len(error)
      request = codec.encode_frame(_catalog_frame(cell, "valid"))
      reply = await cell.handle_frame(request)
      assert codec.decode_frame(reply)["call_id"] == "valid"
      assert cell.budget.used_bytes == len(malformed) + len(error) + len(request) + len(reply)
      assert cell.budget.error is None
    finally:
      cell.close()
  asyncio.run(run())


def test_terminal_error_cannot_echo_oversized_correlation_fields():
  async def run():
    reader = FrameReader()
    # Echoing a near-cap call_id in the reply exceeds the cell limit, forcing
    # _protocol_error_wire to drop correlation fields rather than emit an
    # oversized frame. The request is exactly MAX_CELL_BYTES; its reply cannot fit.
    cell = reader.open_cell("outer-parent", abort_event=None, active=True)
    try:
      frame = {"type": "catalog", "request": "search", "query": "", "call_id": "", "cell_token": cell.token}
      size = len(codec.encode_frame(frame))
      frame["call_id"] = "x" * (codec.MAX_CELL_BYTES - size)
      request = codec.encode_frame(frame)
      assert len(request) == codec.MAX_CELL_BYTES
      wire = await cell.handle_frame(request)
      assert len(wire) < codec.MAX_FRAME_BYTES
      reply = codec.decode_frame(wire)
      assert reply["error"]["code"] == "protocol_error"
      assert reply["call_id"] is None
    finally:
      cell.close()
  asyncio.run(run())
