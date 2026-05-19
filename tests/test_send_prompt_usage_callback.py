import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
PKG_DIR = ROOT / "packages" / "agent-gateway"
if str(PKG_DIR) not in sys.path:
  sys.path.insert(0, str(PKG_DIR))

from agent_gateway.multi_user.billing import UsageEvent
from agent_gateway.send_prompt import _call_usage_callback


def _event() -> UsageEvent:
  return UsageEvent(
    user_id="alice",
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


def test_usage_callback_receives_usage_event() -> None:
  calls: list[UsageEvent] = []

  def cb(event: UsageEvent) -> None:
    calls.append(event)

  _call_usage_callback(cb, _event())
  assert len(calls) == 1
  assert calls[0].input_tokens == 10
