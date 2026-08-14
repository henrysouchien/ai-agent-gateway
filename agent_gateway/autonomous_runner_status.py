from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Any, Callable

from .autonomous_runner_state import AutonomousTask


def tail_lines(log_path: Path, line_count: int) -> tuple[list[str], int]:
  if not log_path.exists():
    return [], 0
  if line_count <= 0:
    total_lines = 0
    with log_path.open("r", encoding="utf-8", errors="replace") as handle:
      for total_lines, _ in enumerate(handle, start=1):
        pass
    return [], total_lines

  total_lines = 0
  recent = deque(maxlen=line_count)
  with log_path.open("r", encoding="utf-8", errors="replace") as handle:
    for total_lines, line in enumerate(handle, start=1):
      recent.append(line.rstrip("\n"))
  return list(recent), total_lines


def status_payload(
  record: AutonomousTask,
  *,
  tail_lines_func: Callable[[Path, int], tuple[list[str], int]] = tail_lines,
  status_tail_lines: int,
) -> dict[str, Any]:
  payload: dict[str, Any] = {
    "state": record.state,
    "elapsed_sec": record.elapsed_sec,
  }
  if record.exit_code is not None:
    payload["exit_code"] = record.exit_code
  if record.error:
    payload["error"] = record.error
  if record.terminal_reason is not None:
    payload["terminal_reason"] = record.terminal_reason
  lines, _total = tail_lines_func(record.log_path, status_tail_lines)
  if lines:
    payload["log_tail"] = "\n".join(lines)
  return payload


__all__ = [
  "status_payload",
  "tail_lines",
]
