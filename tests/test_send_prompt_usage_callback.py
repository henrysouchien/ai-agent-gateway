import logging
import sys
import weakref
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[3]
PKG_DIR = ROOT / "packages" / "agent-gateway"
if str(PKG_DIR) not in sys.path:
  sys.path.insert(0, str(PKG_DIR))

from agent_gateway.multi_user.billing import UsageEvent
from agent_gateway.send_prompt import _call_usage_callback, _classify_usage_callback


def _event() -> UsageEvent:
  return UsageEvent(
    user_id="_default",
    session_id="sess",
    request_id="req",
    parent_turn_id=None,
    timestamp=1.0,
    model="claude-sonnet-4-6",
    input_tokens=10,
    output_tokens=20,
    cache_read_tokens=3,
    cache_creation_tokens=4,
    cost_usd=0.0,
    rate_table_version="unknown",
    billing_mode="byok",
    channel=None,
  )


def test_classify_legacy_four_required_params_warns_once() -> None:
  def cb(a: int, b: int, c: int, d: int, extra: Any = None) -> None:
    _ = a, b, c, d, extra

  with pytest.warns(DeprecationWarning) as warnings:
    assert _classify_usage_callback(cb) == "legacy"
    assert _classify_usage_callback(cb) == "legacy"
  assert len(warnings) == 1


def test_legacy_callback_receives_four_ints() -> None:
  calls: list[tuple[int, int, int, int]] = []

  def cb(a: int, b: int, c: int, d: int) -> None:
    calls.append((a, b, c, d))

  with pytest.warns(DeprecationWarning):
    _call_usage_callback(cb, _event())
  _call_usage_callback(cb, _event())
  assert calls == [(10, 20, 3, 4), (10, 20, 3, 4)]


def test_modern_callback_receives_usage_event_without_warning() -> None:
  calls: list[UsageEvent] = []

  def cb(event: UsageEvent) -> None:
    calls.append(event)

  _call_usage_callback(cb, _event())
  assert len(calls) == 1
  assert calls[0].input_tokens == 10


def test_variadic_callback_treated_as_modern_with_warning(caplog: pytest.LogCaptureFixture) -> None:
  def cb(*args: Any) -> None:
    _ = args

  caplog.set_level(logging.WARNING, logger="agent_gateway.send_prompt")
  assert _classify_usage_callback(cb) == "modern"
  assert "callback uses *args" in caplog.text


@pytest.mark.parametrize("shape", ["two", "five"])
def test_unusual_signature_treated_as_modern_with_warning(
  caplog: pytest.LogCaptureFixture,
  shape: str,
) -> None:
  if shape == "two":
    def cb(a: Any, b: Any) -> None:
      _ = a, b
  else:
    def cb(a: Any, b: Any, c: Any, d: Any, e: Any) -> None:
      _ = a, b, c, d, e

  caplog.set_level(logging.WARNING, logger="agent_gateway.send_prompt")
  assert _classify_usage_callback(cb) == "modern"
  assert "unusual on_usage callback signature" in caplog.text


def test_non_weakref_callable_uses_id_fallback_for_dedup() -> None:
  class Callback:
    __slots__ = ()

    def __call__(self, a: int, b: int, c: int, d: int) -> None:
      _ = a, b, c, d

  cb = Callback()
  with pytest.raises(TypeError):
    weakref.ref(cb)

  with pytest.warns(DeprecationWarning) as warnings:
    assert _classify_usage_callback(cb) == "legacy"
    assert _classify_usage_callback(cb) == "legacy"
  assert len(warnings) == 1

