import asyncio
from types import SimpleNamespace

from agent_gateway.runner_state import (
  append_deferred_tool_extras,
  append_tool_execution_result,
  assistant_turn_message,
  background_tasks_completed_user_message,
  collect_run_agent_batch,
  completed_assistant_content_blocks,
  execute_run_agent_batch,
  execute_run_agent_batch_call,
  execute_run_agent_batch_item,
  execute_run_agent_with_optional_semaphore,
  execute_single_tool_call,
  execute_single_tool_call_step,
  execute_tool_use_loop,
  gather_run_agent_batch_results,
  max_tokens_continuation_decision,
  model_visible_extra_blocks,
  no_tool_use_turn_outcome,
  normalized_run_config,
  process_run_agent_batch_results,
  process_single_tool_result,
  select_run_max_tokens,
  session_drain_state,
  should_execute_run_agent_batch,
  stream_turn_log_summary,
  StreamTurnResult,
  sub_agent_batch_error,
  turn_reminder_state,
  usage_cache_status,
  user_turn_message,
)


def test_normalized_run_config_coerces_fields_and_preserves_extra_keys() -> None:
  auth_config = {
    "auth_mode": " OAuth ",
    "api_key": 123,
    "auth_token": None,
    "model": "",
    "max_tokens": "2048",
    "thinking": "",
    "base_url": "https://example.test",
  }

  config = normalized_run_config(
    auth_config,
    default_model="default-model",
    model_override=None,
  )

  assert config == {
    "auth_mode": "oauth",
    "api_key": "123",
    "auth_token": "None",
    "model": "default-model",
    "max_tokens": 2048,
    "thinking": False,
    "base_url": "https://example.test",
  }
  assert auth_config["auth_mode"] == " OAuth "
  assert auth_config["model"] == ""


def test_normalized_run_config_uses_defaults_for_missing_fields() -> None:
  config = normalized_run_config(
    {},
    default_model="provider-default",
    model_override=None,
  )

  assert config == {
    "auth_mode": "api",
    "api_key": "",
    "auth_token": "",
    "model": "provider-default",
    "max_tokens": 16000,
    "thinking": True,
  }


def test_normalized_run_config_model_override_wins() -> None:
  config = normalized_run_config(
    {"model": "configured-model", "thinking": "false"},
    default_model="provider-default",
    model_override="override-model",
  )

  assert config["model"] == "override-model"
  assert config["thinking"] is True


def test_assistant_turn_message_shapes_conversation_entry_and_copies_content() -> None:
  content = [{"type": "text", "text": "hello"}]

  message = assistant_turn_message(
    content,
    provider="stub",
    model="model-1",
    stop_reason="tool_use",
  )
  content.append({"type": "text", "text": "mutated"})

  assert message == {
    "role": "assistant",
    "content": [{"type": "text", "text": "hello"}],
    "provider": "stub",
    "model": "model-1",
    "stop_reason": "tool_use",
  }


def test_assistant_turn_message_preserves_missing_stop_reason() -> None:
  assert assistant_turn_message(
    [],
    provider="stub",
    model="model-1",
    stop_reason=None,
  ) == {
    "role": "assistant",
    "content": [],
    "provider": "stub",
    "model": "model-1",
    "stop_reason": None,
  }


def test_user_turn_message_shapes_text_content() -> None:
  assert user_turn_message("continue") == {
    "role": "user",
    "content": "continue",
  }


def test_user_turn_message_shapes_list_content_without_copying() -> None:
  content = [{"type": "tool_result", "tool_use_id": "tool-1", "content": "ok"}]

  message = user_turn_message(content)

  assert message == {
    "role": "user",
    "content": content,
  }
  assert message["content"] is content


def test_background_tasks_completed_user_message_shapes_reminder() -> None:
  assert background_tasks_completed_user_message() == {
    "role": "user",
    "content": "[System: Background tasks have completed. Check results with get_background_result.]",
  }


def test_turn_reminder_state_combines_background_and_notification_reminders() -> None:
  state = turn_reminder_state(
    "background",
    "notification",
    pending_notification_count=4,
    max_notifications_per_turn=2,
  )

  assert state.text == "background\n\nnotification"
  assert state.peeked_notification_count == 2


def test_turn_reminder_state_preserves_single_reminders_and_empty_state() -> None:
  assert turn_reminder_state(
    "background",
    "",
    pending_notification_count=4,
    max_notifications_per_turn=2,
  ).text == "background"
  assert turn_reminder_state(
    "background",
    "",
    pending_notification_count=4,
    max_notifications_per_turn=2,
  ).peeked_notification_count == 0
  assert turn_reminder_state(
    "",
    "notification",
    pending_notification_count=1,
    max_notifications_per_turn=5,
  ).text == "notification"
  assert turn_reminder_state(
    "",
    "",
    pending_notification_count=1,
    max_notifications_per_turn=5,
  ).text == ""


def test_completed_assistant_content_blocks_filters_tool_use_blocks() -> None:
  text_block = {"type": "text", "text": "hello"}
  thinking_block = {"type": "thinking", "thinking": "private"}
  tool_use_block = {"type": "tool_use", "id": "tool-1"}
  plain_block = "plain"

  assert completed_assistant_content_blocks([
    text_block,
    tool_use_block,
    thinking_block,
    plain_block,
  ]) == [
    text_block,
    thinking_block,
    plain_block,
  ]


def test_max_tokens_continuation_decision_increments_and_checks_bound() -> None:
  first = max_tokens_continuation_decision(current_attempts=0, max_attempts=2)
  second = max_tokens_continuation_decision(current_attempts=1, max_attempts=2)
  terminal = max_tokens_continuation_decision(current_attempts=2, max_attempts=2)

  assert first.attempts == 1
  assert first.should_continue is True
  assert second.attempts == 2
  assert second.should_continue is True
  assert terminal.attempts == 3
  assert terminal.should_continue is False


def test_no_tool_use_turn_outcome_continues_pause_and_compaction() -> None:
  content = [{"type": "text", "text": "paused"}]

  pause = no_tool_use_turn_outcome(
    content_blocks=content,
    provider="stub",
    model="model-1",
    stop_reason="pause_turn",
    pending_notification_count=0,
    max_tokens_continuations=4,
    max_tokens_max_attempts=5,
    max_tokens_nudge="nudge",
  )
  compaction = no_tool_use_turn_outcome(
    content_blocks=content,
    provider="stub",
    model="model-1",
    stop_reason="compaction",
    pending_notification_count=0,
    max_tokens_continuations=4,
    max_tokens_max_attempts=5,
    max_tokens_nudge="nudge",
  )

  assert pause.action == "continue"
  assert pause.reason == "pause_turn"
  assert pause.messages == [
    {
      "role": "assistant",
      "content": content,
      "provider": "stub",
      "model": "model-1",
      "stop_reason": "pause_turn",
    }
  ]
  assert pause.max_tokens_continuations == 4
  assert pause.runtime_guard is None
  assert compaction.action == "continue"
  assert compaction.reason == "compaction"
  assert compaction.messages[0]["stop_reason"] == "compaction"


def test_no_tool_use_turn_outcome_max_tokens_continues_with_guard_and_nudge() -> None:
  text_block = {"type": "text", "text": "partial"}
  tool_use_block = {"type": "tool_use", "id": "tool-1", "name": "lookup"}

  outcome = no_tool_use_turn_outcome(
    content_blocks=[text_block, tool_use_block],
    provider="stub",
    model="model-1",
    stop_reason="max_tokens",
    pending_notification_count=3,
    max_tokens_continuations=0,
    max_tokens_max_attempts=2,
    max_tokens_nudge="continue please",
  )

  assert outcome.action == "continue"
  assert outcome.reason == "max_tokens_continue"
  assert outcome.max_tokens_continuations == 1
  assert outcome.max_tokens_decision is not None
  assert outcome.max_tokens_decision.should_continue is True
  assert outcome.runtime_guard == ("max_tokens_truncation", "continue please")
  assert outcome.messages == [
    {
      "role": "assistant",
      "content": [text_block],
      "provider": "stub",
      "model": "model-1",
      "stop_reason": "max_tokens",
    },
    {"role": "user", "content": "continue please"},
  ]


def test_no_tool_use_turn_outcome_max_tokens_terminal_breaks_before_notifications() -> None:
  outcome = no_tool_use_turn_outcome(
    content_blocks=[{"type": "text", "text": "partial"}],
    provider="stub",
    model="model-1",
    stop_reason="max_tokens",
    pending_notification_count=3,
    max_tokens_continuations=2,
    max_tokens_max_attempts=2,
    max_tokens_nudge="continue please",
  )

  assert outcome.action == "break"
  assert outcome.reason == "max_tokens_terminal"
  assert outcome.max_tokens_continuations == 3
  assert outcome.max_tokens_decision is not None
  assert outcome.max_tokens_decision.should_continue is False
  assert outcome.messages == []
  assert outcome.runtime_guard is None


def test_no_tool_use_turn_outcome_continues_for_pending_notifications() -> None:
  content = [{"type": "text", "text": "done"}]

  outcome = no_tool_use_turn_outcome(
    content_blocks=content,
    provider="stub",
    model="model-1",
    stop_reason="end_turn",
    pending_notification_count=1,
    max_tokens_continuations=1,
    max_tokens_max_attempts=2,
    max_tokens_nudge="nudge",
  )

  assert outcome.action == "continue"
  assert outcome.reason == "pending_notifications"
  assert outcome.max_tokens_continuations == 1
  assert outcome.messages == [
    {
      "role": "assistant",
      "content": content,
      "provider": "stub",
      "model": "model-1",
      "stop_reason": "end_turn",
    },
    {
      "role": "user",
      "content": "[System: Background tasks have completed. Check results with get_background_result.]",
    },
  ]


def test_no_tool_use_turn_outcome_breaks_default_terminal_turn() -> None:
  outcome = no_tool_use_turn_outcome(
    content_blocks=[{"type": "text", "text": "done"}],
    provider="stub",
    model="model-1",
    stop_reason="end_turn",
    pending_notification_count=0,
    max_tokens_continuations=1,
    max_tokens_max_attempts=2,
    max_tokens_nudge="nudge",
  )

  assert outcome.action == "break"
  assert outcome.reason == "no_tool_use_terminal"
  assert outcome.max_tokens_continuations == 1
  assert outcome.messages == []
  assert outcome.runtime_guard is None


def test_model_visible_extra_blocks_filters_event_only_blocks() -> None:
  visible_text = {"type": "text", "text": "visible"}
  visible_false = {"type": "source_envelope", "_event_only": False, "data": "visible"}
  hidden = {"type": "source_envelope", "_event_only": True, "data": "hidden"}

  assert model_visible_extra_blocks([visible_text, hidden, visible_false]) == [
    visible_text,
    visible_false,
  ]


def test_should_execute_run_agent_batch_requires_run_agent_and_no_exclusion() -> None:
  assert should_execute_run_agent_batch("run_agent", set()) is True
  assert should_execute_run_agent_batch("run_agent", {"run_agent"}) is False
  assert should_execute_run_agent_batch("lookup", set()) is False


def test_collect_run_agent_batch_returns_contiguous_batch_and_next_indexes() -> None:
  first_input = {"task": "a"}
  second_input = {"task": "b"}
  tool_uses = [
    ("tool_1", "lookup", {}),
    ("tool_2", "run_agent", first_input),
    ("tool_3", "run_agent", second_input),
    ("tool_4", "lookup", {}),
  ]
  availability_checks = 0

  def run_agent_available() -> bool:
    nonlocal availability_checks
    availability_checks += 1
    return True

  batch = collect_run_agent_batch(
    tool_uses,
    start_index=1,
    start_call_index=4,
    run_agent_available=run_agent_available,
  )

  assert batch.batch == [
    (1, "tool_2", "run_agent", first_input),
    (2, "tool_3", "run_agent", second_input),
  ]
  assert batch.call_indices == [4, 5]
  assert batch.next_index == 3
  assert batch.next_call_index == 6
  assert availability_checks == 2


def test_collect_run_agent_batch_stops_when_run_agent_becomes_unavailable() -> None:
  tool_uses = [
    ("tool_1", "run_agent", {"task": "a"}),
    ("tool_2", "run_agent", {"task": "b"}),
    ("tool_3", "run_agent", {"task": "c"}),
  ]
  availability = iter([True, False])

  batch = collect_run_agent_batch(
    tool_uses,
    start_index=0,
    start_call_index=0,
    run_agent_available=lambda: next(availability),
  )

  assert batch.batch == [(0, "tool_1", "run_agent", {"task": "a"})]
  assert batch.call_indices == [0]
  assert batch.next_index == 1
  assert batch.next_call_index == 1


def test_collect_run_agent_batch_returns_empty_for_non_run_agent_start() -> None:
  batch = collect_run_agent_batch(
    [("tool_1", "lookup", {})],
    start_index=0,
    start_call_index=3,
    run_agent_available=lambda: True,
  )

  assert batch.batch == []
  assert batch.call_indices == []
  assert batch.next_index == 0
  assert batch.next_call_index == 3


def test_execute_run_agent_with_optional_semaphore_uses_semaphore_for_foreground() -> None:
  events: list[str] = []

  class FakeSemaphore:
    async def __aenter__(self) -> None:
      events.append("enter")

    async def __aexit__(self, exc_type, exc, tb) -> None:
      events.append("exit")

  async def execute() -> str:
    events.append("execute")
    return "done"

  result = asyncio.run(
    execute_run_agent_with_optional_semaphore(
      {"task": "a"},
      sub_agent_semaphore=FakeSemaphore(),
      execute=execute,
    )
  )

  assert result == "done"
  assert events == ["enter", "execute", "exit"]


def test_execute_run_agent_with_optional_semaphore_skips_semaphore_for_background() -> None:
  events: list[str] = []

  class FakeSemaphore:
    async def __aenter__(self) -> None:
      events.append("enter")

    async def __aexit__(self, exc_type, exc, tb) -> None:
      events.append("exit")

  async def execute() -> str:
    events.append("execute")
    return "done"

  result = asyncio.run(
    execute_run_agent_with_optional_semaphore(
      {"background": True},
      sub_agent_semaphore=FakeSemaphore(),
      execute=execute,
    )
  )

  assert result == "done"
  assert events == ["execute"]


def test_execute_run_agent_with_optional_semaphore_runs_without_semaphore() -> None:
  events: list[str] = []

  async def execute() -> str:
    events.append("execute")
    return "done"

  result = asyncio.run(
    execute_run_agent_with_optional_semaphore(
      {"task": "a"},
      sub_agent_semaphore=None,
      execute=execute,
    )
  )

  assert result == "done"
  assert events == ["execute"]


def test_execute_run_agent_batch_item_forwards_call_under_foreground_semaphore() -> None:
  events: list[str] = []
  base_kwargs = {"session": "parent"}
  tool_input = {"task": "a"}
  calls: list[tuple[str, str, dict[str, str], dict[str, str], int]] = []

  class FakeSemaphore:
    async def __aenter__(self) -> None:
      events.append("enter")

    async def __aexit__(self, exc_type, exc, tb) -> None:
      events.append("exit")

  async def execute_single_tool(
    tool_id: str,
    tool_name: str,
    current_tool_input: dict[str, str],
    current_base_kwargs: dict[str, str],
    *,
    call_index: int,
  ) -> str:
    events.append("execute")
    calls.append((tool_id, tool_name, current_tool_input, current_base_kwargs, call_index))
    return "done"

  result = asyncio.run(
    execute_run_agent_batch_item(
      "tool_1",
      "run_agent",
      tool_input,
      7,
      base_kwargs=base_kwargs,
      sub_agent_semaphore=FakeSemaphore(),
      execute_single_tool=execute_single_tool,
    )
  )

  assert result == "done"
  assert events == ["enter", "execute", "exit"]
  assert calls == [("tool_1", "run_agent", tool_input, base_kwargs, 7)]


def test_execute_run_agent_batch_item_skips_semaphore_for_background() -> None:
  events: list[str] = []

  class FakeSemaphore:
    async def __aenter__(self) -> None:
      events.append("enter")

    async def __aexit__(self, exc_type, exc, tb) -> None:
      events.append("exit")

  async def execute_single_tool(*args, call_index: int):
    _ = args, call_index
    events.append("execute")
    return "done"

  result = asyncio.run(
    execute_run_agent_batch_item(
      "tool_1",
      "run_agent",
      {"background": True},
      0,
      base_kwargs={},
      sub_agent_semaphore=FakeSemaphore(),
      execute_single_tool=execute_single_tool,
    )
  )

  assert result == "done"
  assert events == ["execute"]


def test_gather_run_agent_batch_results_forwards_zipped_batch_calls() -> None:
  first_input = {"task": "a"}
  second_input = {"task": "b"}
  batch = [
    (0, "tool_1", "run_agent", first_input),
    (1, "tool_2", "run_agent", second_input),
  ]
  calls: list[tuple[str, str, dict[str, str], int]] = []
  gather_events: list[str] = []

  async def execute(
    tool_id: str,
    tool_name: str,
    tool_input: dict[str, str],
    call_index: int,
  ) -> str:
    calls.append((tool_id, tool_name, tool_input, call_index))
    return f"{tool_id}:{call_index}"

  async def gather_fn(*awaitables, return_exceptions: bool):
    assert return_exceptions is True
    gather_events.append(f"gather:{len(awaitables)}")
    return [await awaitable for awaitable in awaitables]

  results = asyncio.run(
    gather_run_agent_batch_results(
      batch,
      [3, 4],
      execute=execute,
      gather_fn=gather_fn,
    )
  )

  assert results == ["tool_1:3", "tool_2:4"]
  assert calls == [
    ("tool_1", "run_agent", first_input, 3),
    ("tool_2", "run_agent", second_input, 4),
  ]
  assert gather_events == ["gather:2"]


def test_gather_run_agent_batch_results_preserves_zip_truncation() -> None:
  batch = [
    (0, "tool_1", "run_agent", {"task": "a"}),
    (1, "tool_2", "run_agent", {"task": "b"}),
  ]
  calls: list[str] = []

  async def execute(
    tool_id: str,
    tool_name: str,
    tool_input: dict[str, str],
    call_index: int,
  ) -> str:
    _ = tool_name, tool_input
    calls.append(tool_id)
    return f"{tool_id}:{call_index}"

  async def gather_fn(*awaitables, return_exceptions: bool):
    assert return_exceptions is True
    return [await awaitable for awaitable in awaitables]

  results = asyncio.run(
    gather_run_agent_batch_results(
      batch,
      [8],
      execute=execute,
      gather_fn=gather_fn,
    )
  )

  assert results == ["tool_1:8"]
  assert calls == ["tool_1"]


def test_process_run_agent_batch_results_handles_successes_and_exceptions() -> None:
  visible_extra = {"type": "text", "text": "visible"}
  hidden_extra = {"type": "source_envelope", "_event_only": True}
  exc = RuntimeError("boom")
  exception_events: list[str] = []

  def make_error_result(tool_id: str, code: str, message: str) -> dict[str, str]:
    return {"tool_id": tool_id, "code": code, "message": message}

  result = process_run_agent_batch_results(
    [
      (0, "tool_1", "run_agent", {"task": "a"}),
      (1, "tool_2", "run_agent", {"task": "b"}),
    ],
    [
      ({"type": "tool_result", "tool_use_id": "tool_1"}, "run_agent", [visible_extra, hidden_extra]),
      exc,
    ],
    batch_error=sub_agent_batch_error,
    make_error_result=make_error_result,
    model_visible_extra_blocks=model_visible_extra_blocks,
    on_exception=lambda error: exception_events.append(str(error)),
  )

  assert result.tool_results_content == [
    {"type": "tool_result", "tool_use_id": "tool_1"},
    {"tool_id": "tool_2", "code": "sub_agent_error", "message": "boom"},
  ]
  assert result.deferred_extras == [visible_extra]
  assert result.tools_used == ["run_agent", "run_agent"]
  assert exception_events == ["boom"]


def test_process_run_agent_batch_results_preserves_batch_index_lookup() -> None:
  calls: list[str] = []

  result = process_run_agent_batch_results(
    [
      (4, "tool_5", "run_agent", {"task": "e"}),
      (7, "tool_8", "run_agent", {"task": "h"}),
    ],
    [asyncio.CancelledError()],
    batch_error=sub_agent_batch_error,
    make_error_result=lambda tool_id, code, message: {
      "tool_id": tool_id,
      "code": code,
      "message": message,
    },
    model_visible_extra_blocks=model_visible_extra_blocks,
    on_exception=lambda error: calls.append(type(error).__name__),
  )

  assert result.tool_results_content == [
    {"tool_id": "tool_5", "code": "cancelled", "message": "Sub-agent was cancelled"}
  ]
  assert result.deferred_extras == []
  assert result.tools_used == ["run_agent"]
  assert calls == ["CancelledError"]


def test_process_single_tool_result_filters_deferred_extras_and_tracks_tool() -> None:
  result_entry = {"type": "tool_result", "tool_use_id": "tool_1"}
  visible_extra = {"type": "text", "text": "visible"}
  hidden_extra = {"type": "source_envelope", "_event_only": True}

  result = process_single_tool_result(
    result_entry,
    "lookup",
    [visible_extra, hidden_extra],
    model_visible_extra_blocks=model_visible_extra_blocks,
  )

  assert result.tool_results_content == [result_entry]
  assert result.deferred_extras == [visible_extra]
  assert result.tools_used == ["lookup"]


def test_append_tool_execution_result_extends_existing_aggregates() -> None:
  existing_result = {"type": "tool_result", "tool_use_id": "tool_0"}
  result_entry = {"type": "tool_result", "tool_use_id": "tool_1"}
  visible_extra = {"type": "text", "text": "visible"}
  tool_results_content = [existing_result]
  deferred_extras = [{"type": "text", "text": "previous"}]
  tools_used = ["previous"]

  append_tool_execution_result(
    tool_results_content,
    deferred_extras,
    tools_used,
    process_single_tool_result(
      result_entry,
      "lookup",
      [visible_extra],
      model_visible_extra_blocks=model_visible_extra_blocks,
    ),
  )

  assert tool_results_content == [existing_result, result_entry]
  assert deferred_extras == [{"type": "text", "text": "previous"}, visible_extra]
  assert tools_used == ["previous", "lookup"]


def test_append_deferred_tool_extras_extends_results_after_tool_results() -> None:
  first_result = {"type": "tool_result", "tool_use_id": "tool_1"}
  visible_extra = {"type": "text", "text": "visible"}
  second_extra = {"type": "text", "text": "second"}
  tool_results_content = [first_result]
  deferred_extras = [visible_extra, second_extra]

  append_deferred_tool_extras(tool_results_content, deferred_extras)

  assert tool_results_content == [first_result, visible_extra, second_extra]
  assert deferred_extras == [visible_extra, second_extra]


def test_execute_single_tool_call_forwards_arguments_and_processes_result() -> None:
  result_entry = {"type": "tool_result", "tool_use_id": "tool_1"}
  tool_input = {"query": "value"}
  base_kwargs = {"session_id": "sid-1"}
  visible_extra = {"type": "text", "text": "visible"}
  hidden_extra = {"type": "source_envelope", "_event_only": True}
  calls: list[tuple[str, str, dict[str, str], dict[str, str]]] = []

  async def execute_single_tool(
    tool_id: str,
    tool_name: str,
    payload: dict[str, str],
    kwargs: dict[str, str],
  ):
    calls.append((tool_id, tool_name, payload, kwargs))
    return result_entry, "lookup", [visible_extra, hidden_extra]

  result = asyncio.run(
    execute_single_tool_call(
      "tool_1",
      "lookup",
      tool_input,
      base_kwargs=base_kwargs,
      execute_single_tool=execute_single_tool,
      model_visible_extra_blocks=model_visible_extra_blocks,
    )
  )

  assert calls == [("tool_1", "lookup", tool_input, base_kwargs)]
  assert calls[0][2] is tool_input
  assert calls[0][3] is base_kwargs
  assert result.tool_results_content == [result_entry]
  assert result.deferred_extras == [visible_extra]
  assert result.tools_used == ["lookup"]


def test_execute_single_tool_call_step_processes_result_and_advances_index() -> None:
  result_entry = {"type": "tool_result", "tool_use_id": "tool_1"}
  tool_input = {"query": "value"}
  base_kwargs = {"session_id": "sid-1"}
  visible_extra = {"type": "text", "text": "visible"}
  hidden_extra = {"type": "source_envelope", "_event_only": True}
  calls: list[tuple[str, str, dict[str, str], dict[str, str]]] = []

  async def execute_single_tool(
    tool_id: str,
    tool_name: str,
    payload: dict[str, str],
    kwargs: dict[str, str],
  ):
    calls.append((tool_id, tool_name, payload, kwargs))
    return result_entry, "lookup", [visible_extra, hidden_extra]

  result = asyncio.run(
    execute_single_tool_call_step(
      "tool_1",
      "lookup",
      tool_input,
      current_index=4,
      base_kwargs=base_kwargs,
      execute_single_tool=execute_single_tool,
      model_visible_extra_blocks=model_visible_extra_blocks,
    )
  )

  assert calls == [("tool_1", "lookup", tool_input, base_kwargs)]
  assert result.next_index == 5
  assert result.tool_result.tool_results_content == [result_entry]
  assert result.tool_result.deferred_extras == [visible_extra]
  assert result.tool_result.tools_used == ["lookup"]


def test_execute_run_agent_batch_gathers_and_processes_results() -> None:
  visible_extra = {"type": "text", "text": "visible"}
  hidden_extra = {"type": "source_envelope", "_event_only": True}
  exception_events: list[str] = []
  execute_calls: list[tuple[str, int]] = []

  async def execute(
    tool_id: str,
    tool_name: str,
    tool_input: dict[str, str],
    call_index: int,
  ):
    _ = tool_name, tool_input
    execute_calls.append((tool_id, call_index))
    if tool_id == "tool_2":
      raise RuntimeError("boom")
    return ({"type": "tool_result", "tool_use_id": tool_id}, "run_agent", [visible_extra, hidden_extra])

  async def gather_fn(*awaitables, return_exceptions: bool):
    assert return_exceptions is True
    results = []
    for awaitable in awaitables:
      try:
        results.append(await awaitable)
      except BaseException as exc:
        results.append(exc)
    return results

  result = asyncio.run(
    execute_run_agent_batch(
      [
        (0, "tool_1", "run_agent", {"task": "a"}),
        (1, "tool_2", "run_agent", {"task": "b"}),
      ],
      [5, 6],
      execute=execute,
      gather_fn=gather_fn,
      batch_error=sub_agent_batch_error,
      make_error_result=lambda tool_id, code, message: {
        "tool_id": tool_id,
        "code": code,
        "message": message,
      },
      model_visible_extra_blocks=model_visible_extra_blocks,
      on_exception=lambda error: exception_events.append(str(error)),
    )
  )

  assert result.tool_results_content == [
    {"type": "tool_result", "tool_use_id": "tool_1"},
    {"tool_id": "tool_2", "code": "sub_agent_error", "message": "boom"},
  ]
  assert result.deferred_extras == [visible_extra]
  assert result.tools_used == ["run_agent", "run_agent"]
  assert execute_calls == [("tool_1", 5), ("tool_2", 6)]
  assert exception_events == ["boom"]


def test_execute_run_agent_batch_call_collects_executes_and_returns_next_state() -> None:
  visible_extra = {"type": "text", "text": "visible"}
  hidden_extra = {"type": "source_envelope", "_event_only": True}
  availability_checks: list[str] = []
  exception_events: list[str] = []
  execute_calls: list[tuple[str, int]] = []

  async def execute(
    tool_id: str,
    tool_name: str,
    tool_input: dict[str, str],
    call_index: int,
  ):
    _ = tool_name, tool_input
    execute_calls.append((tool_id, call_index))
    return ({"type": "tool_result", "tool_use_id": tool_id}, "run_agent", [visible_extra, hidden_extra])

  async def gather_fn(*awaitables, return_exceptions: bool):
    assert return_exceptions is True
    return [await awaitable for awaitable in awaitables]

  def run_agent_available() -> bool:
    availability_checks.append("checked")
    return True

  result = asyncio.run(
    execute_run_agent_batch_call(
      [
        ("tool_1", "run_agent", {"task": "a"}),
        ("tool_2", "run_agent", {"task": "b"}),
        ("tool_3", "lookup", {"query": "c"}),
      ],
      start_index=0,
      start_call_index=7,
      run_agent_available=run_agent_available,
      execute=execute,
      gather_fn=gather_fn,
      batch_error=sub_agent_batch_error,
      make_error_result=lambda tool_id, code, message: {
        "tool_id": tool_id,
        "code": code,
        "message": message,
      },
      model_visible_extra_blocks=model_visible_extra_blocks,
      on_exception=lambda error: exception_events.append(str(error)),
    )
  )

  assert result.next_index == 2
  assert result.next_call_index == 9
  assert result.batch_result.tool_results_content == [
    {"type": "tool_result", "tool_use_id": "tool_1"},
    {"type": "tool_result", "tool_use_id": "tool_2"},
  ]
  assert result.batch_result.deferred_extras == [visible_extra, visible_extra]
  assert result.batch_result.tools_used == ["run_agent", "run_agent"]
  assert execute_calls == [("tool_1", 7), ("tool_2", 8)]
  assert availability_checks == ["checked", "checked"]
  assert exception_events == []


def test_execute_tool_use_loop_processes_batches_and_single_tools_in_order() -> None:
  visible_extra = {"type": "text", "text": "visible"}
  hidden_extra = {"type": "source_envelope", "_event_only": True}
  base_kwargs = {"session_id": "sid-1"}
  excluded_tool_checks: list[str] = []
  exception_events: list[str] = []
  execute_calls: list[tuple[str, str, int | None]] = []

  async def execute_single_tool(
    tool_id: str,
    tool_name: str,
    tool_input: dict[str, str],
    kwargs: dict[str, str],
    **extra_kwargs,
  ):
    assert kwargs is base_kwargs
    _ = tool_input
    execute_calls.append((tool_id, tool_name, extra_kwargs.get("call_index")))
    return (
      {"type": "tool_result", "tool_use_id": tool_id},
      tool_name,
      [visible_extra, hidden_extra],
    )

  async def gather_fn(*awaitables, return_exceptions: bool):
    assert return_exceptions is True
    return [await awaitable for awaitable in awaitables]

  def effective_excluded_tools() -> set[str]:
    excluded_tool_checks.append("checked")
    return set()

  result = asyncio.run(
    execute_tool_use_loop(
      [
        ("tool_1", "run_agent", {"task": "a"}),
        ("tool_2", "run_agent", {"task": "b"}),
        ("tool_3", "lookup", {"query": "c"}),
        ("tool_4", "run_agent", {"task": "d"}),
      ],
      base_kwargs=base_kwargs,
      sub_agent_semaphore=None,
      effective_excluded_tools=effective_excluded_tools,
      execute_single_tool=execute_single_tool,
      gather_fn=gather_fn,
      batch_error=sub_agent_batch_error,
      make_error_result=lambda tool_id, code, message: {
        "tool_id": tool_id,
        "code": code,
        "message": message,
      },
      model_visible_extra_blocks=model_visible_extra_blocks,
      on_exception=lambda error: exception_events.append(str(error)),
    )
  )

  assert result.tool_results_content == [
    {"type": "tool_result", "tool_use_id": "tool_1"},
    {"type": "tool_result", "tool_use_id": "tool_2"},
    {"type": "tool_result", "tool_use_id": "tool_3"},
    {"type": "tool_result", "tool_use_id": "tool_4"},
    visible_extra,
    visible_extra,
    visible_extra,
    visible_extra,
  ]
  assert result.tools_used == ["run_agent", "run_agent", "lookup", "run_agent"]
  assert execute_calls == [
    ("tool_1", "run_agent", 0),
    ("tool_2", "run_agent", 1),
    ("tool_3", "lookup", None),
    ("tool_4", "run_agent", 2),
  ]
  assert len(excluded_tool_checks) >= 4
  assert exception_events == []


def test_execute_tool_use_loop_treats_excluded_run_agent_as_single_tool() -> None:
  visible_extra = {"type": "text", "text": "visible"}
  hidden_extra = {"type": "source_envelope", "_event_only": True}
  execute_calls: list[tuple[str, str, int | None]] = []

  async def execute_single_tool(
    tool_id: str,
    tool_name: str,
    tool_input: dict[str, str],
    kwargs: dict[str, str],
    **extra_kwargs,
  ):
    assert kwargs == {"session_id": "sid-1"}
    assert tool_input == {"task": "a"}
    execute_calls.append((tool_id, tool_name, extra_kwargs.get("call_index")))
    return (
      {"type": "tool_result", "tool_use_id": tool_id},
      tool_name,
      [visible_extra, hidden_extra],
    )

  async def gather_fn(*_awaitables, return_exceptions: bool):
    raise AssertionError("excluded run_agent should not enter batch gather")

  def on_exception(error: BaseException) -> None:
    raise AssertionError("excluded run_agent should not report batch exceptions") from error

  result = asyncio.run(
    execute_tool_use_loop(
      [("tool_1", "run_agent", {"task": "a"})],
      base_kwargs={"session_id": "sid-1"},
      sub_agent_semaphore=None,
      effective_excluded_tools=lambda: {"run_agent"},
      execute_single_tool=execute_single_tool,
      gather_fn=gather_fn,
      batch_error=sub_agent_batch_error,
      make_error_result=lambda tool_id, code, message: {
        "tool_id": tool_id,
        "code": code,
        "message": message,
      },
      model_visible_extra_blocks=model_visible_extra_blocks,
      on_exception=on_exception,
    )
  )

  assert result.tool_results_content == [
    {"type": "tool_result", "tool_use_id": "tool_1"},
    visible_extra,
  ]
  assert result.tools_used == ["run_agent"]
  assert execute_calls == [("tool_1", "run_agent", None)]


def test_sub_agent_batch_error_maps_cancelled_error() -> None:
  error = sub_agent_batch_error(asyncio.CancelledError())

  assert error.code == "cancelled"
  assert error.message == "Sub-agent was cancelled"


def test_sub_agent_batch_error_maps_regular_exception_and_empty_message() -> None:
  assert sub_agent_batch_error(RuntimeError("boom")).code == "sub_agent_error"
  assert sub_agent_batch_error(RuntimeError("boom")).message == "boom"
  assert sub_agent_batch_error(RuntimeError()).message == "Sub-agent failed"


def test_session_drain_state_counts_only_unfinished_asyncio_tasks() -> None:
  pending_task = SimpleNamespace(done=lambda: False)
  done_task = SimpleNamespace(done=lambda: True)

  state = session_drain_state(
    [
      SimpleNamespace(asyncio_task=None),
      SimpleNamespace(asyncio_task=done_task),
      SimpleNamespace(asyncio_task=pending_task),
    ],
    shutdown_failed=False,
  )

  assert state.in_flight_task_count == 1
  assert state.drain_complete is False


def test_session_drain_state_requires_no_shutdown_failure() -> None:
  assert session_drain_state([], shutdown_failed=False).drain_complete is True
  failed = session_drain_state([], shutdown_failed=True)

  assert failed.in_flight_task_count == 0
  assert failed.drain_complete is False


def test_usage_cache_status_formats_read_write_and_miss() -> None:
  assert usage_cache_status({
    "cache_read_input_tokens": 12,
    "cache_creation_input_tokens": 7,
  }) == "hit (12 tokens cached)"
  assert usage_cache_status({
    "cache_read_input_tokens": 0,
    "cache_creation_input_tokens": 7,
  }) == "write (7 tokens written)"
  assert usage_cache_status({
    "cache_read_input_tokens": 0,
    "cache_creation_input_tokens": 0,
  }) == "miss"


def test_stream_turn_log_summary_extracts_logging_fields() -> None:
  turn = StreamTurnResult(
    full_text=("hello\n" * 40),
    tool_uses=[
      ("tool-1", "search", {}),
      ("tool-2", "write", {}),
    ],
    stop_reason="tool_use",
    first_token_t=12.5,
  )

  summary = stream_turn_log_summary(turn, turn_started_at=10.0, now=16.25)

  assert summary.elapsed_s == 6.25
  assert summary.ttft_s == 2.5
  assert summary.text_chars == 240
  assert summary.tool_names == ["search", "write"]
  assert summary.text_preview == turn.full_text[:150].replace("\n", " ")


def test_stream_turn_log_summary_handles_empty_text_and_missing_first_token() -> None:
  summary = stream_turn_log_summary(
    StreamTurnResult(),
    turn_started_at=10.0,
    now=11.0,
  )

  assert summary.elapsed_s == 1.0
  assert summary.ttft_s is None
  assert summary.text_chars == 0
  assert summary.tool_names == []
  assert summary.text_preview == ""


def test_select_run_max_tokens_uses_config_and_override() -> None:
  model_info = SimpleNamespace(max_output_tokens=0)

  assert select_run_max_tokens(
    {"max_tokens": 16000},
    max_tokens_override=None,
    model_info=model_info,
  ).value == 16000
  selection = select_run_max_tokens(
    {"max_tokens": 16000},
    max_tokens_override=32000,
    model_info=model_info,
  )

  assert selection.value == 32000
  assert selection.requested == 32000
  assert selection.model_max_output == 0
  assert selection.clamped is False


def test_select_run_max_tokens_clamps_to_model_max_output() -> None:
  selection = select_run_max_tokens(
    {"max_tokens": 64000},
    max_tokens_override=None,
    model_info=SimpleNamespace(max_output_tokens=16384),
  )

  assert selection.value == 16384
  assert selection.requested == 64000
  assert selection.model_max_output == 16384
  assert selection.clamped is True
