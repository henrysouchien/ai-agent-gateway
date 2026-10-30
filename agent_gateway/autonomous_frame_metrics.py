from __future__ import annotations

from dataclasses import dataclass
from threading import Lock
from typing import Literal

from pydantic import BaseModel, ConfigDict


class FrameCapFinding(BaseModel):
  model_config = ConfigDict(frozen=True, strict=True, extra="forbid")

  channel: Literal["claim_broker", "event_channel"]
  direction: Literal["send", "receive"]
  size_basis: Literal["actual", "declared", "lower_bound"]
  frame_size_bytes: int
  cap_bytes: int


class FrameCapFindingEvent(BaseModel):
  model_config = ConfigDict(strict=True, extra="forbid")

  type: Literal["frame_cap_finding"]
  finding: FrameCapFinding


@dataclass(frozen=True)
class FrameCapSnapshot:
  count: int
  last_finding: FrameCapFinding | None


class FrameCapMetrics:
  def __init__(self, channel: Literal["claim_broker", "event_channel"]) -> None:
    self._channel: Literal["claim_broker", "event_channel"] = channel
    self._lock = Lock()
    self._count = 0
    self._last_finding: FrameCapFinding | None = None

  def record(
    self,
    *,
    frame_size_bytes: int,
    cap_bytes: int,
    direction: Literal["send", "receive"],
    size_basis: Literal["actual", "declared", "lower_bound"],
  ) -> FrameCapFinding:
    finding = FrameCapFinding(
      channel=self._channel,
      direction=direction,
      size_basis=size_basis,
      frame_size_bytes=frame_size_bytes,
      cap_bytes=cap_bytes,
    )
    return self.record_finding(finding)

  def record_finding(self, finding: FrameCapFinding) -> FrameCapFinding:
    with self._lock:
      self._count += 1
      self._last_finding = finding
    return finding

  def snapshot(self) -> FrameCapSnapshot:
    with self._lock:
      return FrameCapSnapshot(
        count=self._count,
        last_finding=self._last_finding,
      )


__all__ = [
  "FrameCapFinding",
  "FrameCapFindingEvent",
  "FrameCapSnapshot",
  "FrameCapMetrics",
]
