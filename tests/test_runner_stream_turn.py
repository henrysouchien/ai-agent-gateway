import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
PKG_DIR = ROOT / "packages" / "agent-gateway"
if str(PKG_DIR) not in sys.path:
  sys.path.insert(0, str(PKG_DIR))

from agent_gateway import AgentRunner  # noqa: E402
from agent_gateway.providers import ModelInfo, ThinkingLevel  # noqa: E402
import agent_gateway.runner as gateway_runner  # noqa: E402
from agent_gateway.runner_stream_turn import RunnerStreamTurnMixin  # noqa: E402


def test_runner_stream_turn_methods_are_inherited_from_mixin() -> None:
  assert issubclass(AgentRunner, RunnerStreamTurnMixin)
  assert gateway_runner.RunnerStreamTurnMixin is RunnerStreamTurnMixin

  for method_name in (
    "_thinking_level",
    "_effective_stream_stall_timeout",
    "_classify_guard_outcome",
    "_stream_turn",
  ):
    assert getattr(AgentRunner, method_name) is getattr(RunnerStreamTurnMixin, method_name)


def test_stream_turn_helpers_resolve_parent_module_aliases(monkeypatch) -> None:
  runner = object.__new__(AgentRunner)
  runner._stream_stall_timeout = None
  calls: dict[str, object] = {}

  monkeypatch.setattr(gateway_runner, "STREAM_STALL_TIMEOUT", 12.0)
  monkeypatch.setattr(gateway_runner, "STREAM_THINKING_STALL_TIMEOUT", 34.0)
  monkeypatch.setattr(gateway_runner, "thinking_level", lambda enabled: f"patched-{enabled}")

  def _effective_stream_stall_timeout(override, **kwargs):
    calls["override"] = override
    calls["kwargs"] = kwargs
    return 56.0

  def _classify_guard_outcome(guard_reason, attempt, max_attempts):
    calls["guard"] = (guard_reason, attempt, max_attempts)
    return ("patched", "guard", "kind")

  monkeypatch.setattr(gateway_runner, "effective_stream_stall_timeout", _effective_stream_stall_timeout)
  monkeypatch.setattr(gateway_runner, "classify_guard_outcome", _classify_guard_outcome)

  model_info = ModelInfo(id="model", provider="stub", supports_thinking=True)

  assert AgentRunner._thinking_level(True) == "patched-True"
  assert (
    runner._effective_stream_stall_timeout(
      config={"thinking": True},
      model_info=model_info,
      max_tokens=128,
    )
    == 56.0
  )
  assert AgentRunner._classify_guard_outcome(("stall", "quiet"), 1, 3) == ("patched", "guard", "kind")

  assert calls["kwargs"] == {
    "config": {"thinking": True},
    "model_info": model_info,
    "max_tokens": 128,
    "observed_thinking": False,
    "stream_stall_timeout_default": 12.0,
    "stream_thinking_stall_timeout_default": 34.0,
  }
  assert calls["guard"] == (("stall", "quiet"), 1, 3)


def test_runner_stream_turn_reexports_streaming_helpers() -> None:
  assert gateway_runner.AgentRunner._thinking_level(False) is ThinkingLevel.NONE
  assert gateway_runner.STREAM_STALL_TIMEOUT > 0
  assert gateway_runner.STREAM_THINKING_STALL_TIMEOUT > gateway_runner.STREAM_STALL_TIMEOUT
