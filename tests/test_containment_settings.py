"""``ContainmentSettings`` resolves once and every bound reads it."""

import asyncio
import math
import sys
import time
from pathlib import Path
from types import MethodType, SimpleNamespace
from typing import Any

import pytest

TESTS_DIR = Path(__file__).resolve().parent
if str(TESTS_DIR) not in sys.path:
  sys.path.insert(0, str(TESTS_DIR))

import agent_gateway.runner_sub_agents as runner_sub_agents  # noqa: E402
from agent_gateway.admission import AdmittedContext, Refusal, admit  # noqa: E402
from agent_gateway.child_environment import (  # noqa: E402
  AUTONOMOUS_CHILD_ENVIRONMENT,
  filter_child_environment,
)
from agent_gateway.containment_settings import (  # noqa: E402
  CONTAINMENT_ENV,
  CONTAINMENT_ENV_NAMES,
  ContainmentSettings,
  resolve_containment_settings,
)
from agent_gateway.event_log import EventLog  # noqa: E402
from agent_gateway.runner import AgentRunner, StreamTurnResult  # noqa: E402
from agent_gateway.runner_soft_budget import SoftBudgetLadder  # noqa: E402
from gateway_test_support.capability_execution_test_support import (  # noqa: E402
  stub_runner_capability_execution,
)
from gateway_test_support.runner_dispatcher_test_support import (  # noqa: E402
  CataloglessDispatcherTestSupport,
)
from test_admission import _PlainResolver, _owners, _request, _root  # noqa: E402


def test_defaults_are_the_ratified_values() -> None:
  settings = ContainmentSettings()

  assert settings.max_spawn_depth == 2
  assert settings.max_concurrent_children == 6
  assert dict(settings.soft_turn_budget) == {"default": 100, "autonomous": 100, "fork": 10}
  assert settings.soft_turn_budget_cap is None
  assert settings.soft_budget_final_factor == 1.5
  assert settings.soft_budget_grace_turns == 2
  assert settings.child_silence_timeout_seconds == 600.0
  assert settings.model_context_limit == 200_000
  assert settings.context_warning_pct == 80
  assert settings.context_pressure_reminder_pct == 60
  assert settings.context_pressure_reminder_step_pct == 10
  assert settings.compaction_trigger_pct == 25


@pytest.mark.parametrize(("setting", "env"), [
  (setting, env) for setting, (env, _parse) in CONTAINMENT_ENV.items()
])
def test_each_env_name_parses_into_its_setting(setting: str, env: str) -> None:
  parse = CONTAINMENT_ENV[setting][1]
  raw = "2.5" if parse is float else "7"

  settings = resolve_containment_settings({env: raw})

  assert getattr(settings, setting) == parse(raw)
  defaults = ContainmentSettings()
  for other in CONTAINMENT_ENV:
    if other != setting:
      assert getattr(settings, other) == getattr(defaults, other)


def test_unset_or_blank_env_keeps_the_defaults() -> None:
  assert resolve_containment_settings({}) == ContainmentSettings()
  assert resolve_containment_settings(
    {env: "  " for env in CONTAINMENT_ENV_NAMES},
  ) == ContainmentSettings()


@pytest.mark.parametrize(("env", "raw", "setting"), [
  ("HANK_MAX_SPAWN_DEPTH", "lots", "max_spawn_depth"),
  ("HANK_MAX_SPAWN_DEPTH", "2.5", "max_spawn_depth"),
  ("HANK_SOFT_TURN_BUDGET", "lots", "soft_turn_budget_cap"),
  ("HANK_SOFT_BUDGET_FINAL_FACTOR", "steep", "soft_budget_final_factor"),
  ("HANK_SOFT_BUDGET_FINAL_FACTOR", "0.5", "soft_budget_final_factor"),
  ("HANK_SOFT_BUDGET_GRACE_TURNS", "-1", "soft_budget_grace_turns"),
  ("HANK_CHILD_SILENCE_TIMEOUT_SECONDS", "-1", "child_silence_timeout_seconds"),
  ("HANK_SOFT_BUDGET_FINAL_FACTOR", "nan", "soft_budget_final_factor"),
  ("HANK_SOFT_BUDGET_FINAL_FACTOR", "inf", "soft_budget_final_factor"),
  ("HANK_CHILD_SILENCE_TIMEOUT_SECONDS", "nan", "child_silence_timeout_seconds"),
  ("HANK_CHILD_SILENCE_TIMEOUT_SECONDS", "inf", "child_silence_timeout_seconds"),
  ("HANK_CONTEXT_WARNING_PCT", "101", "context_warning_pct"),
])
def test_invalid_values_warn_naming_the_value_and_keep_the_default(
  env: str, raw: str, setting: str, caplog: pytest.LogCaptureFixture,
) -> None:
  with caplog.at_level("WARNING", logger="agent_gateway.containment_settings"):
    settings = resolve_containment_settings({env: raw})

  assert settings == ContainmentSettings()
  assert getattr(settings, setting) == getattr(ContainmentSettings(), setting)
  warnings = [record.getMessage() for record in caplog.records]
  assert len(warnings) == 1
  assert setting in warnings[0] and env in warnings[0] and raw in warnings[0]


def test_unreadable_soft_turn_budget_leaves_the_table_in_force() -> None:
  settings = resolve_containment_settings({"HANK_SOFT_TURN_BUDGET": "lots"})

  assert settings.soft_turn_budget_cap is None
  assert settings.soft_turn_threshold("sub_agent") == 100
  assert settings.soft_turn_threshold("fork") == 10


def test_zero_child_silence_timeout_is_the_off_value() -> None:
  settings = resolve_containment_settings({"HANK_CHILD_SILENCE_TIMEOUT_SECONDS": "0"})

  assert settings.child_silence_timeout_seconds == 0.0


def _spawn_owners(settings: ContainmentSettings):
  return _owners(
    _PlainResolver(),
    ceiling=frozenset({"file_read", "run_agent"}),
    children=("explore",),
    containment=settings,
  )


@pytest.mark.parametrize("maximum", [1, 2, 4])
def test_spawn_depth_bound_comes_from_the_admission_owners(maximum: int) -> None:
  owners = _spawn_owners(ContainmentSettings(max_spawn_depth=maximum))

  deepest = admit(_request(), _root(depth=maximum - 1), owners=owners)
  assert isinstance(deepest, AdmittedContext)
  assert deepest.wire.lineage.depth == maximum
  assert deepest.wire.tool_grant == ("file_read",)

  refused = admit(_request(), _root(depth=maximum), owners=owners)
  assert isinstance(refused, Refusal)
  assert refused.reason == "depth_exceeded"
  assert refused.classification == "resource"


@pytest.mark.parametrize("parent_depth", [0, 2, 9])
def test_negative_spawn_depth_env_disables_the_bound(parent_depth: int) -> None:
  settings = resolve_containment_settings({"HANK_MAX_SPAWN_DEPTH": "-1"})

  result = admit(
    _request(), _root(depth=parent_depth), owners=_spawn_owners(settings),
  )

  assert isinstance(result, AdmittedContext)
  assert result.wire.lineage.depth == parent_depth + 1
  assert result.wire.tool_grant == ("file_read", "run_agent")


class _FakeProvider:
  name = "fake"

  def has_active_credential(self, config: dict[str, Any]) -> bool:
    return True

  def create_client(self, config: dict[str, Any], *, timeout: float | None = None) -> object:
    return object()

  async def close_client(self, client: object, timeout: float = 2.0) -> None:
    return None

  def get_model_info(self, model: str) -> SimpleNamespace:
    return SimpleNamespace(
      id=model,
      provider="fake",
      context_window=200_000,
      max_output_tokens=8192,
      supports_thinking=False,
    )

  def estimate_cost(self, model: str, input_tokens: int, output_tokens: int, **_kwargs: Any):
    return SimpleNamespace(total=0.0)


class _RecordingDispatcher(CataloglessDispatcherTestSupport):
  async def dispatch(
    self,
    tool_call_id: str,
    tool_name: str,
    tool_input: dict[str, Any],
    *,
    call_index: int = 0,
    allow_uncertain_mcp_replay: bool = False,
    on_executed_prepared_call: Any | None = None,
  ):
    return {"ok": True}, None


def _runner(containment: ContainmentSettings) -> AgentRunner:
  return AgentRunner(
    event_log=EventLog(),
    dispatcher=_RecordingDispatcher(),  # type: ignore[arg-type]
    session_id="containment-settings",
    capability_execution=stub_runner_capability_execution(
      provider=_FakeProvider(),
      model="claude-sonnet-5",
      effort="none",
      auth_config={"auth_mode": "api", "api_key": "x", "max_tokens": 1024},
    ),
    user_id="alice",
    billing_mode="byok",
    rate_table_version="unknown",
    containment=containment,
  )


def test_runner_queue_width_follows_the_settings() -> None:
  runner = _runner(ContainmentSettings(max_concurrent_children=3))

  assert runner._max_background_tasks == 3
  semaphore = runner._ensure_sub_agent_semaphore()
  assert isinstance(semaphore, asyncio.Semaphore)
  assert semaphore._value == 3


def test_soft_ladder_final_stage_follows_the_factor() -> None:
  ladder = SoftBudgetLadder(turn_threshold=4, final_factor=2.25, grace_turns=1)

  fired = {
    turns: action.stage
    for turns in range(0, 20)
    if (action := ladder.observe(turn_count=turns)) is not None
  }

  assert fired == {4: "steer", math.ceil(4 * 2.25): "final", 10: "abort"}


def test_runner_ladder_follows_the_cap_factor_and_grace() -> None:
  runner = _runner(ContainmentSettings(
    soft_turn_budget_cap=2, soft_budget_final_factor=2.0, soft_budget_grace_turns=1,
  ))
  runner._soft_budget_kind = "sub_agent"
  requests = 0

  async def _tool_every_turn(self, *, client: Any, usage_totals: dict[str, Any], **_kwargs: Any):
    nonlocal requests
    requests += 1
    if requests > 20:
      raise AssertionError("the soft-budget ladder never ended the run")
    usage_totals["capability_bind"] = self.capability_execution.bind.to_json()
    usage_totals["provider_reported_model"] = None
    tool_id = f"tool_{requests}"
    return client, StreamTurnResult(
      stop_reason="tool_use",
      tool_uses=[(tool_id, "record_tool", {})],
      content_blocks=[{"type": "tool_use", "id": tool_id, "name": "record_tool", "input": {}}],
    )

  runner._stream_turn = MethodType(_tool_every_turn, runner)
  asyncio.run(runner.run(messages=[{"role": "user", "content": "go"}], system_prompt="x"))

  ladder = [
    (entry.event["type"], entry.event.get("observed"), entry.event.get("threshold"))
    for entry in runner._log.entries
    if str(entry.event.get("type") or "").startswith("soft_budget")
  ]
  assert ladder == [("soft_budget_steer", 2, 2), ("soft_budget_final", 4, 2)]
  # The forced final opened before request 5; one grace request, then abort.
  assert requests == 5


def test_zero_child_silence_timeout_never_cancels(monkeypatch: pytest.MonkeyPatch) -> None:
  monkeypatch.setattr(runner_sub_agents, "STREAM_GUARD_POLL_INTERVAL", 0.01)

  async def case(silence_timeout: float) -> tuple[float | None, bool]:
    activity = runner_sub_agents._ChildActivity()
    activity.last_at = time.monotonic() - 10_000
    finished = False

    async def child() -> None:
      nonlocal finished
      await asyncio.sleep(0.05)
      finished = True

    silence = await runner_sub_agents._await_child_under_activity_guard(
      child(), activity, silence_timeout=silence_timeout,
    )
    return silence, finished

  assert asyncio.run(case(0)) == (None, True)
  silence, finished = asyncio.run(case(1.0))
  assert silence is not None and silence > 1.0
  assert finished is False


def test_autonomous_child_projection_carries_every_containment_name() -> None:
  environ = {name: "1" for name in CONTAINMENT_ENV_NAMES}

  projected = filter_child_environment(
    {**environ, "UNLISTED_SECRET": "x"},
    purpose=AUTONOMOUS_CHILD_ENVIRONMENT,
  )

  assert projected == environ
