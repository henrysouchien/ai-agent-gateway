import asyncio
import sys
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[3]
PKG_DIR = ROOT / "packages" / "agent-gateway"
if str(PKG_DIR) not in sys.path:
  sys.path.insert(0, str(PKG_DIR))

import agent_gateway.retry as retry
from agent_gateway import RetryConfig, RunOutput, classify_outcome, run_autonomous_with_retry


def _run(coro):
  return asyncio.run(coro)


def _output(
  response: str = "ok",
  *,
  error: str | None = None,
  timed_out: bool = False,
  budget_exceeded: bool = False,
  max_turns_reached: bool = False,
  operator_paused: bool = False,
) -> RunOutput:
  return RunOutput(
    response=response,
    tools_used=[],
    usage={},
    error=error,
    timed_out=timed_out,
    budget_exceeded=budget_exceeded,
    max_turns_reached=max_turns_reached,
    operator_paused=operator_paused,
  )


def _sequence_run_fn(items: list[RunOutput | Exception], calls: list[dict[str, Any]] | None = None):
  remaining = list(items)

  async def _run_fn(*args: Any, **kwargs: Any) -> RunOutput:
    if calls is not None:
      calls.append({"args": args, "kwargs": kwargs})
    item = remaining.pop(0)
    if isinstance(item, Exception):
      raise item
    return item

  return _run_fn


class _StatusError(Exception):
  def __init__(self, status_code: int, message: str = "status error") -> None:
    super().__init__(message)
    self.status_code = status_code


class APIConnectionError(Exception):
  pass


class APITimeoutError(Exception):
  pass


class TransportError(Exception):
  pass


class StreamError(Exception):
  pass


class _ValueError503(ValueError):
  def __init__(self) -> None:
    super().__init__("service unavailable")
    self.status_code = 503


def test_run_autonomous_with_retry_success_first_attempt(monkeypatch: pytest.MonkeyPatch) -> None:
  sleep_calls: list[float] = []
  call_args: list[dict[str, Any]] = []

  async def _fake_sleep(delay: float) -> None:
    sleep_calls.append(delay)

  monkeypatch.setattr(retry.asyncio, "sleep", _fake_sleep)

  payload = {"x": 1}
  output = _run(
    run_autonomous_with_retry(
      _sequence_run_fn([_output("done")], calls=call_args),
      payload,
      flag=True,
    )
  )

  assert output.response == "done"
  assert len(call_args) == 1
  assert call_args[0]["args"] == (payload,)
  assert call_args[0]["kwargs"] == {"flag": True}
  assert sleep_calls == []


def test_run_autonomous_with_retry_transient_then_success(monkeypatch: pytest.MonkeyPatch) -> None:
  sleep_calls: list[float] = []

  async def _fake_sleep(delay: float) -> None:
    sleep_calls.append(delay)

  monkeypatch.setattr(retry.asyncio, "sleep", _fake_sleep)

  run_fn = _sequence_run_fn([
    _output(error="HTTP 503 Service Unavailable"),
    _output("recovered"),
  ])

  output = _run(
    run_autonomous_with_retry(
      run_fn,
      retry_config=RetryConfig(max_retries=2, backoff_steps=[1.5]),
    )
  )

  assert output.response == "recovered"
  assert sleep_calls == [1.5]


def test_run_autonomous_with_retry_permanent_auth_error(monkeypatch: pytest.MonkeyPatch) -> None:
  sleep_calls: list[float] = []

  async def _fake_sleep(delay: float) -> None:
    sleep_calls.append(delay)

  monkeypatch.setattr(retry.asyncio, "sleep", _fake_sleep)

  output = _run(
    run_autonomous_with_retry(
      _sequence_run_fn([_output(error="HTTP 401 Unauthorized")]),
      retry_config=RetryConfig(max_retries=3, backoff_steps=[1.0]),
    )
  )

  assert output.error == "HTTP 401 Unauthorized"
  assert sleep_calls == []


def test_run_autonomous_with_retry_budget_exceeded_is_permanent(monkeypatch: pytest.MonkeyPatch) -> None:
  sleep_calls: list[float] = []

  async def _fake_sleep(delay: float) -> None:
    sleep_calls.append(delay)

  monkeypatch.setattr(retry.asyncio, "sleep", _fake_sleep)

  output = _run(
    run_autonomous_with_retry(
      _sequence_run_fn([_output(budget_exceeded=True)]),
      retry_config=RetryConfig(max_retries=3),
    )
  )

  assert output.budget_exceeded is True
  assert sleep_calls == []


def test_run_autonomous_with_retry_max_turns_reached_is_permanent(monkeypatch: pytest.MonkeyPatch) -> None:
  sleep_calls: list[float] = []

  async def _fake_sleep(delay: float) -> None:
    sleep_calls.append(delay)

  monkeypatch.setattr(retry.asyncio, "sleep", _fake_sleep)

  output = _run(
    run_autonomous_with_retry(
      _sequence_run_fn([_output(max_turns_reached=True)]),
      retry_config=RetryConfig(max_retries=3),
    )
  )

  assert output.max_turns_reached is True
  assert sleep_calls == []


def test_run_autonomous_with_retry_operator_pause_is_not_retried(monkeypatch: pytest.MonkeyPatch) -> None:
  sleep_calls: list[float] = []

  async def _fake_sleep(delay: float) -> None:
    sleep_calls.append(delay)

  monkeypatch.setattr(retry.asyncio, "sleep", _fake_sleep)

  output = _run(
    run_autonomous_with_retry(
      _sequence_run_fn([_output(operator_paused=True), _output("unused")]),
      retry_config=RetryConfig(max_retries=3),
    )
  )

  assert output.operator_paused is True
  assert sleep_calls == []


def test_run_autonomous_with_retry_timeout_retries_when_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
  sleep_calls: list[float] = []

  async def _fake_sleep(delay: float) -> None:
    sleep_calls.append(delay)

  monkeypatch.setattr(retry.asyncio, "sleep", _fake_sleep)

  output = _run(
    run_autonomous_with_retry(
      _sequence_run_fn([_output(timed_out=True), _output("after-timeout")]),
      retry_config=RetryConfig(max_retries=1, backoff_steps=[2.0], retry_on_timeout=True),
    )
  )

  assert output.response == "after-timeout"
  assert sleep_calls == [2.0]


def test_run_autonomous_with_retry_timeout_does_not_retry_when_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
  sleep_calls: list[float] = []

  async def _fake_sleep(delay: float) -> None:
    sleep_calls.append(delay)

  monkeypatch.setattr(retry.asyncio, "sleep", _fake_sleep)

  output = _run(
    run_autonomous_with_retry(
      _sequence_run_fn([_output(timed_out=True), _output("unused")]),
      retry_config=RetryConfig(max_retries=1, backoff_steps=[2.0], retry_on_timeout=False),
    )
  )

  assert output.timed_out is True
  assert output.response == "ok"
  assert sleep_calls == []


def test_run_autonomous_with_retry_returns_last_output_when_max_retries_exhausted(monkeypatch: pytest.MonkeyPatch) -> None:
  sleep_calls: list[float] = []

  async def _fake_sleep(delay: float) -> None:
    sleep_calls.append(delay)

  monkeypatch.setattr(retry.asyncio, "sleep", _fake_sleep)

  items = [
    _output(error="HTTP 503 Service Unavailable"),
    _output(error="HTTP 503 Service Unavailable"),
    _output(error="HTTP 503 Service Unavailable"),
    _output(error="HTTP 503 Service Unavailable"),
  ]

  output = _run(
    run_autonomous_with_retry(
      _sequence_run_fn(items),
      retry_config=RetryConfig(max_retries=3, backoff_steps=[1.0, 2.0]),
    )
  )

  assert output.error == "HTTP 503 Service Unavailable"
  assert sleep_calls == [1.0, 2.0, 2.0]


def test_run_autonomous_with_retry_retries_transient_exceptions(monkeypatch: pytest.MonkeyPatch) -> None:
  sleep_calls: list[float] = []

  async def _fake_sleep(delay: float) -> None:
    sleep_calls.append(delay)

  monkeypatch.setattr(retry.asyncio, "sleep", _fake_sleep)

  output = _run(
    run_autonomous_with_retry(
      _sequence_run_fn([APIConnectionError("dial failed"), _output("after-exception")]),
      retry_config=RetryConfig(max_retries=1, backoff_steps=[0.25]),
    )
  )

  assert output.response == "after-exception"
  assert sleep_calls == [0.25]


def test_run_autonomous_with_retry_invokes_on_retry_callback(monkeypatch: pytest.MonkeyPatch) -> None:
  sleep_calls: list[float] = []
  callback_calls: list[tuple[int, str, float]] = []

  async def _fake_sleep(delay: float) -> None:
    sleep_calls.append(delay)

  async def _on_retry(attempt_number: int, error_description: str, delay: float) -> None:
    callback_calls.append((attempt_number, error_description, delay))

  monkeypatch.setattr(retry.asyncio, "sleep", _fake_sleep)

  output = _run(
    run_autonomous_with_retry(
      _sequence_run_fn([_output(error="HTTP 503 Service Unavailable"), _output("ok")]),
      retry_config=RetryConfig(max_retries=1, backoff_steps=[3.0], on_retry=_on_retry),
    )
  )

  assert output.response == "ok"
  assert callback_calls == [(2, "HTTP 503 Service Unavailable", 3.0)]
  assert sleep_calls == [3.0]


def test_classify_outcome_error_string_patterns() -> None:
  assert classify_outcome(None, _output(error="HTTP 429 Too Many Requests")) == "transient"
  assert classify_outcome(None, _output(error="529 overloaded")) == "transient"
  assert classify_outcome(None, _output(error="Connection reset by peer")) == "transient"
  assert classify_outcome(None, _output(error="HTTP 401 Unauthorized")) == "permanent"
  assert classify_outcome(None, _output(error="Bad request: malformed input")) == "permanent"
  assert classify_outcome(None, _output(error="FileNotFoundError: missing config")) == "permanent"


def test_classify_outcome_exception_types_and_status_codes() -> None:
  assert classify_outcome(_StatusError(401), None) == "permanent"
  assert classify_outcome(_StatusError(429), None) == "transient"
  assert classify_outcome(_StatusError(503), None) == "transient"
  assert classify_outcome(APIConnectionError("dial failed"), None) == "transient"
  assert classify_outcome(APITimeoutError("slow"), None) == "transient"
  assert classify_outcome(TransportError("network"), None) == "transient"
  assert classify_outcome(StreamError("stream"), None) == "transient"
  assert classify_outcome(ConnectionError("socket"), None) == "transient"
  assert classify_outcome(ValueError("bad config"), None) == "permanent"
  assert classify_outcome(TypeError("bad type"), None) == "permanent"
  assert classify_outcome(FileNotFoundError("missing"), None) == "permanent"
  assert classify_outcome(KeyError("missing"), None) == "permanent"
  assert classify_outcome(AttributeError("missing"), None) == "permanent"
  assert classify_outcome(_ValueError503(), None) == "transient"
