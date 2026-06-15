from __future__ import annotations

from agent_gateway import approval_settings


def test_approval_wait_seconds_defaults_clamps_and_falls_back(
  monkeypatch,
  caplog,
) -> None:
  monkeypatch.delenv(approval_settings.APPROVAL_WAIT_SECONDS_ENV, raising=False)
  assert approval_settings.approval_wait_seconds() == 270.0

  monkeypatch.setenv(approval_settings.APPROVAL_WAIT_SECONDS_ENV, "5")
  assert approval_settings.approval_wait_seconds() == 10.0

  monkeypatch.setenv(approval_settings.APPROVAL_WAIT_SECONDS_ENV, "2000")
  assert approval_settings.approval_wait_seconds() == 1800.0

  monkeypatch.setenv(approval_settings.APPROVAL_WAIT_SECONDS_ENV, "45.5")
  assert approval_settings.approval_wait_seconds() == 45.5

  with caplog.at_level("WARNING", logger="agent_gateway.approval_settings"):
    monkeypatch.setenv(approval_settings.APPROVAL_WAIT_SECONDS_ENV, "garbage")
    assert approval_settings.approval_wait_seconds() == 270.0
  assert "Invalid GATEWAY_APPROVAL_WAIT_SECONDS" in caplog.text


def test_approval_wait_seconds_reads_env_each_call(monkeypatch) -> None:
  monkeypatch.setenv(approval_settings.APPROVAL_WAIT_SECONDS_ENV, "45")
  assert approval_settings.approval_wait_seconds() == 45.0

  monkeypatch.setenv(approval_settings.APPROVAL_WAIT_SECONDS_ENV, "30")
  assert approval_settings.approval_wait_seconds() == 30.0


def test_approval_wait_seconds_wires_dispatcher_and_sdk_runner_runtime(
  monkeypatch,
) -> None:
  from agent_gateway import sdk_runner, tool_dispatcher

  monkeypatch.setenv(approval_settings.APPROVAL_WAIT_SECONDS_ENV, "45")
  assert tool_dispatcher._approval_queue_timeout_seconds(600) == 45.0
  assert sdk_runner._approval_queue_timeout_seconds(600) == 45.0

  monkeypatch.setenv(approval_settings.APPROVAL_WAIT_SECONDS_ENV, "30")
  assert tool_dispatcher._approval_queue_timeout_seconds(600) == 30.0
  assert sdk_runner._approval_queue_timeout_seconds(600) == 30.0
