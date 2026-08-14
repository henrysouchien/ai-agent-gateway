from __future__ import annotations

import json
import os
from pathlib import Path
import signal
from types import SimpleNamespace

import pytest

from agent_gateway import autonomous_control_files
from agent_gateway import autonomous_runner
from agent_gateway.autonomous_control_files import (
  AutonomousControlAppendError,
  append_closed_json_record,
  secure_create_owned_file,
)


def _payload() -> dict[str, object]:
  return {
    "version": 1,
    "kind": "operator_message",
    "task_id": "bg_1",
    "control_run_id": "run_1",
    "session_id": "session_1",
    "channel_id": "a" * 64,
    "message_id": "message_1",
    "text": "check exposure",
    "sent_at_ns": 1,
  }


def _create(path: Path) -> os.stat_result:
  fd, identity = secure_create_owned_file(path)
  os.close(fd)
  return identity


def test_control_append_rolls_back_fsync_failure(
  tmp_path: Path,
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  path = tmp_path / "operator.jsonl"
  identity = _create(path)
  real_fsync = os.fsync
  calls = 0

  def _fail_first_fsync(fd: int) -> None:
    nonlocal calls
    calls += 1
    if calls == 1:
      raise OSError("injected append fsync failure")
    real_fsync(fd)

  monkeypatch.setattr(
    autonomous_control_files.os,
    "fsync",
    _fail_first_fsync,
  )

  with pytest.raises(AutonomousControlAppendError) as caught:
    append_closed_json_record(
      path,
      expected_device=identity.st_dev,
      expected_inode=identity.st_ino,
      payload=_payload(),
    )

  assert caught.value.stream_recovered is True
  assert path.read_bytes() == b""


def test_control_append_rolls_back_partial_write(
  tmp_path: Path,
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  path = tmp_path / "operator.jsonl"
  identity = _create(path)
  real_write = os.write

  def _partial_write(fd: int, payload: bytes) -> int:
    return real_write(fd, payload[: len(payload) // 2])

  monkeypatch.setattr(
    autonomous_control_files.os,
    "write",
    _partial_write,
  )

  with pytest.raises(AutonomousControlAppendError) as caught:
    append_closed_json_record(
      path,
      expected_device=identity.st_dev,
      expected_inode=identity.st_ino,
      payload=_payload(),
    )

  assert caught.value.stream_recovered is True
  assert path.read_bytes() == b""


def test_control_append_reports_unrecoverable_stream(
  tmp_path: Path,
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  path = tmp_path / "operator.jsonl"
  identity = _create(path)

  def _fail_write(_fd: int, _payload: bytes) -> int:
    raise OSError("injected write failure")

  def _fail_truncate(_fd: int, _size: int) -> None:
    raise OSError("injected rollback failure")

  monkeypatch.setattr(
    autonomous_control_files.os,
    "write",
    _fail_write,
  )
  monkeypatch.setattr(
    autonomous_control_files.os,
    "ftruncate",
    _fail_truncate,
  )

  with pytest.raises(AutonomousControlAppendError) as caught:
    append_closed_json_record(
      path,
      expected_device=identity.st_dev,
      expected_inode=identity.st_ino,
      payload=_payload(),
    )

  assert caught.value.stream_recovered is False


def test_registry_fences_unrecoverable_control_stream(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  registry = object.__new__(autonomous_runner.AutonomousRegistry)
  record = SimpleNamespace(
    cancellation_requested=False,
    error=None,
    proc=SimpleNamespace(returncode=None),
  )
  lifeline_closed: list[object] = []
  signals: list[tuple[object, signal.Signals]] = []

  def _fail_append(*_args: object, **_kwargs: object) -> None:
    raise AutonomousControlAppendError(
      "injected",
      stream_recovered=False,
    )

  monkeypatch.setattr(
    autonomous_runner,
    "_append_control_record",
    _fail_append,
  )
  monkeypatch.setattr(
    registry,
    "_close_owner_lifeline",
    lambda value: lifeline_closed.append(value),
  )
  monkeypatch.setattr(
    registry,
    "_signal_owned_process_group",
    lambda value, sent_signal: signals.append((
      value,
      sent_signal,
    )),
  )

  with pytest.raises(AutonomousControlAppendError):
    registry._append_control_record_or_fence(
      record,
      endpoint="operator_inbox",
      payload=_payload(),
    )

  assert record.cancellation_requested is True
  assert record.error == (
    "Unrecoverable autonomous operator_inbox append failure"
  )
  assert lifeline_closed == [record]
  assert signals == [(record, signal.SIGTERM)]


def test_control_append_success_is_one_closed_record(
  tmp_path: Path,
) -> None:
  path = tmp_path / "operator.jsonl"
  identity = _create(path)

  append_closed_json_record(
    path,
    expected_device=identity.st_dev,
    expected_inode=identity.st_ino,
    payload=_payload(),
  )

  raw = path.read_bytes()
  assert raw.endswith(b"\n")
  assert json.loads(raw) == _payload()
