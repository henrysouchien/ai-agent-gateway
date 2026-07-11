import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
PKG_DIR = ROOT / "packages" / "agent-gateway"
if str(PKG_DIR) not in sys.path:
  sys.path.insert(0, str(PKG_DIR))

from agent_gateway.providers import ModelInfo, ThinkingLevel  # noqa: E402
import agent_gateway.runner as gateway_runner  # noqa: E402
from agent_gateway.runner_streaming import (  # noqa: E402
  STREAM_STALL_TIMEOUT,
  STREAM_THINKING_STALL_TIMEOUT,
  classify_guard_outcome,
  effective_stream_stall_timeout,
  observed_thinking_in_messages,
  thinking_level,
)


def test_runner_reexports_streaming_constants() -> None:
  assert gateway_runner.STREAM_STALL_TIMEOUT == STREAM_STALL_TIMEOUT
  assert gateway_runner.STREAM_THINKING_STALL_TIMEOUT == STREAM_THINKING_STALL_TIMEOUT


def test_thinking_level_maps_boolean_to_provider_enum() -> None:
  assert thinking_level(True) is ThinkingLevel.HIGH
  assert thinking_level(False) is ThinkingLevel.NONE
  assert gateway_runner.AgentRunner._thinking_level(True) is ThinkingLevel.HIGH


def test_effective_stream_stall_timeout_respects_override_and_thinking_defaults() -> None:
  thinking_model = ModelInfo(id="claude-opus-4-7", provider="anthropic", supports_thinking=True)
  plain_model = ModelInfo(id="gpt-test", provider="openai", supports_thinking=False)

  assert (
    effective_stream_stall_timeout(None, config={"thinking": True}, model_info=thinking_model, max_tokens=4096)
    == STREAM_THINKING_STALL_TIMEOUT
  )
  assert effective_stream_stall_timeout(42, config={"thinking": True}, model_info=thinking_model, max_tokens=4096) == 42
  assert (
    effective_stream_stall_timeout(None, config={"thinking": False}, model_info=thinking_model, max_tokens=4096)
    == STREAM_STALL_TIMEOUT
  )
  assert (
    effective_stream_stall_timeout(None, config={"thinking": True}, model_info=plain_model, max_tokens=4096)
    == STREAM_STALL_TIMEOUT
  )
  assert (
    effective_stream_stall_timeout(None, config={"thinking": True}, model_info=thinking_model, max_tokens=1024)
    == STREAM_STALL_TIMEOUT
  )
  assert (
    effective_stream_stall_timeout(
      None,
      config={"thinking": False},
      model_info=thinking_model,
      max_tokens=4096,
      observed_thinking=True,
    )
    == STREAM_THINKING_STALL_TIMEOUT
  )


def test_observed_thinking_in_messages_is_scoped_to_current_model() -> None:
  model_info = ModelInfo(id="claude-sonnet-5", provider="anthropic", supports_thinking=True)
  matching = {
    "role": "assistant",
    "provider": "anthropic",
    "model": "claude-sonnet-5",
    "content": [{"type": "thinking", "thinking": "", "signature": "sig"}],
  }

  assert observed_thinking_in_messages([matching], model_info=model_info) is True
  assert observed_thinking_in_messages(
    [{**matching, "model": "claude-opus-4-8"}],
    model_info=model_info,
  ) is False
  assert observed_thinking_in_messages(
    [{**matching, "provider": "openai"}],
    model_info=model_info,
  ) is False
  assert observed_thinking_in_messages(
    [{**matching, "content": [{"type": "text", "text": "done"}]}],
    model_info=model_info,
  ) is False


def test_classify_guard_outcome_distinguishes_retry_abort_and_non_guard() -> None:
  assert classify_guard_outcome(None, attempt=1, max_attempts=3) == ("not_guard", "", "")
  assert classify_guard_outcome(("stall", "quiet"), attempt=1, max_attempts=3) == (
    "retry",
    "Stream watchdog: quiet",
    "stall",
  )
  assert classify_guard_outcome(("stall", "quiet"), attempt=3, max_attempts=3) == (
    "abort",
    "Stream watchdog: quiet",
    "stall",
  )
  assert classify_guard_outcome(("timeout", "slow"), attempt=1, max_attempts=3) == (
    "abort",
    "Stream watchdog: slow",
    "timeout",
  )
  assert gateway_runner.AgentRunner._classify_guard_outcome(("stall", "quiet"), 1, 2) == (
    "retry",
    "Stream watchdog: quiet",
    "stall",
  )
