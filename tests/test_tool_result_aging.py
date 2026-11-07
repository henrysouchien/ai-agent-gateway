"""L4: old, large data tool results age out of the request into a stub ``tool_result_read`` resolves."""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any

import pytest

PKG_DIR = Path(__file__).resolve().parents[1]
if str(PKG_DIR) not in sys.path:
  sys.path.insert(0, str(PKG_DIR))

from agent_gateway.tool_dispatcher_helpers import LocalToolHandler
from agent_gateway import AgentRunner, AgentSessionLog, EventLog, ToolDispatcher  # noqa: E402
from agent_gateway.admission import ParentContext  # noqa: E402
from agent_gateway.approval_route import NoApprovalRoute  # noqa: E402
from agent_gateway.containment_settings import ContainmentSettings  # noqa: E402
from agent_gateway.context_builder import SessionContextBuilder  # noqa: E402
from agent_gateway.mcp_client import McpClientManager  # noqa: E402
from agent_gateway.registry_scope import RegistryScope  # noqa: E402
from agent_gateway.run_settings import RunSettingsError, pinned_run_settings  # noqa: E402
from agent_gateway.tool_result_aging import (  # noqa: E402
  INSTRUCTION_RESULT_TOOLS,
  ToolResultAging,
  age_tool_results,
  build_aging_stub,
  h1_identifiers,
)
from agent_gateway.tool_result_spill import (  # noqa: E402
  SpillCapabilities,
  SpillSink,
  make_tool_result_read_handler,
)
from agent_workflow_contracts.models import AdmittedResearchBinding, AdmittedTenant  # noqa: E402
from gateway_test_support.capability_execution_test_support import (  # noqa: E402
  stub_runner_capability_execution,
)
from gateway_test_support.run_context_test_support import run_context_for_test  # noqa: E402
from hank_model_transport.providers import ModelInfo, ModelProvider, StreamEvent  # noqa: E402


# ---- the pure projection -------------------------------------------------


def _assistant(text: str = "ok") -> dict[str, Any]:
  return {"role": "assistant", "content": [{"type": "text", "text": text}]}


def _result(tool_use_id: str, content: str) -> dict[str, Any]:
  return {
    "role": "user",
    "content": [{"type": "tool_result", "tool_use_id": tool_use_id, "content": content}],
  }


def _history(replies_after_first_result: int) -> list[dict[str, Any]]:
  messages = [{"role": "user", "content": "go"}, _assistant(), _result("r1", "FULL-1")]
  for index in range(replies_after_first_result):
    messages += [_assistant(str(index)), _result(f"x{index}", "FULL-x")]
  return messages


def _contents(messages: list[dict[str, Any]]) -> dict[str, Any]:
  return {
    block["tool_use_id"]: block["content"]
    for message in messages
    if isinstance(message.get("content"), list)
    for block in message["content"]
    if block.get("type") == "tool_result"
  }


AGING = ToolResultAging(after_replies=4, min_chars=10, step_replies=3)


def _aged_ids(replies_after_first_result: int) -> set[str]:
  history = _history(replies_after_first_result)
  stubs = {tool_use_id: f"STUB-{tool_use_id}" for tool_use_id in _contents(history)}
  aged = age_tool_results(history, aging=AGING, stubs=stubs)
  return {
    tool_use_id for tool_use_id, content in _contents(aged).items()
    if content == f"STUB-{tool_use_id}"
  }


def test_the_aging_boundary_moves_only_at_a_step() -> None:
  # Totals of assistant replies 1..9 (history(k) holds k + 1 replies).
  aged = {replies + 1: _aged_ids(replies) for replies in range(0, 9)}
  assert all(not ids for total, ids in aged.items() if total < 6)
  assert aged[6]
  # Between two steps the aged set, hence the projected prefix, does not change.
  assert aged[6] == aged[7] == aged[8]
  # At the next step the boundary advances and takes more results.
  assert aged[8] < aged[9]
  # An aged result is always older than every result that stays whole.
  assert "r1" in aged[6]


def test_aging_never_mutates_its_input_and_shares_untouched_messages() -> None:
  history = _history(8)
  snapshot = json.dumps(history)
  aged = age_tool_results(history, aging=AGING, stubs={"r1": "STUB-1"})
  assert json.dumps(history) == snapshot
  assert aged is not history
  assert _contents(aged)["r1"] == "STUB-1"
  assert aged[2] is not history[2]
  assert all(new is old for new, old in zip(aged[3:], history[3:]))
  assert age_tool_results(history, aging=AGING, stubs={}) is history
  assert age_tool_results(history, aging=AGING, stubs={"unknown": "S"}) is history


def test_aging_is_the_input_object_until_enough_replies_follow() -> None:
  history = _history(1)
  assert age_tool_results(history, aging=AGING, stubs={"r1": "STUB-1"}) is history


def test_stub_is_deterministic_and_carries_head_identifiers_and_ref() -> None:
  content = json.dumps({"rows": [{"ticker": "AAPL-2024", "ref": f"doc_{i:05d}"} for i in range(50)]})
  first = build_aging_stub(content, tool_name="corpus_state", spill_ref="spill:v1:x:abc", head_chars=100)
  assert first == build_aging_stub(content, tool_name="corpus_state", spill_ref="spill:v1:x:abc", head_chars=100)
  stub = json.loads(first)
  assert stub["_aged_tool_result"] is True
  assert stub["tool_name"] == "corpus_state"
  assert stub["spill_ref"] == "spill:v1:x:abc"
  assert stub["original_chars"] == len(content)
  assert stub["head"] == content[:100]
  assert "doc_00049" in stub["identifiers"] and "AAPL-2024" in stub["identifiers"]
  assert stub["identifiers"] == h1_identifiers(content)


def test_identifiers_are_the_h1_set_longest_first() -> None:
  tokens = h1_identifiers("a_b c1 plain words 123456 12,345,678 abcde fgh-ij x:y:z!")
  assert "123456" in tokens and "12345678" in tokens and "fgh-ij" in tokens
  assert "plain" not in tokens and "abcde" not in tokens and "c1" not in tokens
  assert h1_identifiers("zz_1234 aa_1234567 bb_12345678", limit=2) == ["bb_12345678", "aa_1234567"]


# ---- settings ------------------------------------------------------------


def test_settings_default_off_parse_and_refuse_zero() -> None:
  defaults = pinned_run_settings()
  assert defaults.tool_result_aging_after_replies is None
  assert defaults.tool_result_aging_min_chars == 8000
  assert defaults.tool_result_aging_step_replies == 8
  assert pinned_run_settings(tool_result_aging_after_replies=8).tool_result_aging_after_replies == 8
  assert ContainmentSettings(tool_result_aging_after_replies=8).tool_result_aging_after_replies == 8
  assert ContainmentSettings(tool_result_aging_min_chars=5).tool_result_aging_after_replies is None
  for key in (
    "tool_result_aging_after_replies",
    "tool_result_aging_min_chars",
    "tool_result_aging_step_replies",
  ):
    with pytest.raises(RunSettingsError):
      pinned_run_settings(**{key: 0})


# ---- the receipt, the request and the restore through a real runner -----------


DATA = json.dumps({"rows": [{"id": f"row_{i:05d}", "v": "x" * 20} for i in range(120)]})
SKILL_BODY = "METHOD " * 800


class _ScriptedProvider(ModelProvider):
  name = "stub"

  def __init__(self, script: list[tuple[str, str] | None]) -> None:
    # (tool_use_id, tool_name) per tool turn; None is the closing text turn.
    self._script = list(script)
    self.requests: list[list[dict[str, Any]]] = []

  def has_active_credential(self, config: dict[str, Any]) -> bool:
    return True

  def create_client(self, config: dict[str, Any], *, timeout: float | None = None) -> Any:
    return object()

  async def close_client(self, client: Any, timeout: float = 2.0) -> None:
    pass

  def get_model_info(self, model: str) -> ModelInfo:
    return ModelInfo(id=model, provider=self.name, max_output_tokens=16_384)

  def build_request_params(
    self,
    *,
    model: str,
    messages: list[dict[str, Any]],
    system_prompt: Any,
    tools: list[dict[str, Any]],
    max_tokens: int,
    **kwargs: Any,
  ) -> dict[str, Any]:
    # `messages` can be the live history object: keep the bytes this request carried.
    self.requests.append(json.loads(json.dumps(messages, default=str)))
    return {
      "model": model, "messages": messages, "system_prompt": system_prompt,
      "tools": tools, "max_tokens": max_tokens, **kwargs,
    }

  async def stream(self, client: Any, params: dict[str, Any]):
    yield StreamEvent(type="message_start", input_tokens=10)
    step = self._script.pop(0)
    if step is None:
      yield StreamEvent(type="text_delta", text="done")
      yield StreamEvent(type="text_end", raw_block={"type": "text", "text": "done"})
      yield StreamEvent(type="message_end", stop_reason="end_turn")
      return
    tool_id, tool_name = step
    yield StreamEvent(
      type="tool_use_end", tool_id=tool_id, tool_name=tool_name, tool_input={},
      raw_block={"type": "tool_use", "id": tool_id, "name": tool_name, "input": {}},
    )
    yield StreamEvent(type="message_end", stop_reason="tool_use")


def _definition(name: str) -> dict[str, Any]:
  return {"name": name, "description": "", "input_schema": {"type": "object", "properties": {}}}


def _reader() -> ParentContext:
  return ParentContext(
    tenant=AdmittedTenant(tenant_id=None, storage_id="alice", role="owner"),
    research_binding=AdmittedResearchBinding(),
    children=None,
    tool_ids=None,
    withheld=frozenset(),
    handle_grants=frozenset(),
    remaining_usd=None,
    job_id="reader",
    depth=0,
    approval_route=NoApprovalRoute(),
  )


async def _short(_input: dict[str, Any], **_: Any) -> tuple[Any, Any]:
  return {"status": "ok"}, None


async def _data(_input: dict[str, Any], **_: Any) -> tuple[Any, Any]:
  return json.loads(DATA), None


async def _skill(_input: dict[str, Any], **_: Any) -> tuple[Any, Any]:
  return {"body": SKILL_BODY}, None


async def _drive(
  tmp_path: Path,
  script: list[tuple[str, str] | None],
  *,
  after_replies: int | None,
  step_replies: int = 1,
  log: AgentSessionLog | None = None,
  context_builder: SessionContextBuilder | None = None,
  prompt: str = "go",
) -> tuple[_ScriptedProvider, AgentSessionLog, SpillSink, AgentRunner]:
  spill_root = tmp_path / "spills"
  spill_root.mkdir(parents=True, exist_ok=True)
  sink = SpillSink(
    lambda: str(spill_root),
    capabilities=SpillCapabilities(code_execute=True, spill_read=True),
  )
  log = log or AgentSessionLog(tmp_path / "session.jsonl")
  handlers: dict[str, LocalToolHandler] = {
    "big_data": _data, "small": _short, "invoke_skill": _skill, "docs_fetch": _skill,
  }
  provider = _ScriptedProvider(script)
  event_log = EventLog()
  capability_execution = stub_runner_capability_execution(
    provider=provider, auth_config={"api_key": "k"}, model="stub-model", effort="none",
  )
  dispatcher = ToolDispatcher(
    capability_bind=capability_execution.bind,
    mcp_client=McpClientManager(config_path=None),
    local_tool_handlers=handlers,
    event_log=event_log,
    session_id="sess-aging",
    role="owner",
    run_context=run_context_for_test(registry_scope=RegistryScope("run", "aging-run", "/tmp/aging")),
  )
  overrides: dict[str, Any] = {
    "tool_result_aging_min_chars": 1000,
    "tool_result_aging_step_replies": step_replies,
  }
  if after_replies is not None:
    overrides["tool_result_aging_after_replies"] = after_replies
  runner = AgentRunner(
    event_log=event_log,
    dispatcher=dispatcher,
    session_id="sess-aging",
    capability_execution=capability_execution,
    get_tool_definitions=lambda: [_definition(name) for name in handlers],
    user_id="alice",
    request_id="req-aging",
    billing_mode="byok",
    rate_table_version="unknown",
    agent_session_log=log,
    context_builder=context_builder,
    code_execution_spill_dir_provider=sink,
    containment=ContainmentSettings(**overrides),
  )
  await runner.run(messages=[{"role": "user", "content": prompt}], system_prompt="x")
  return provider, log, sink, runner


SCRIPT: list[tuple[str, str] | None] = [
  ("d1", "big_data"),
  ("s1", "invoke_skill"),
  ("f1", "docs_fetch"),
  ("t1", "small"),
  ("t2", "small"),
  ("t3", "small"),
  ("t4", "small"),
  None,
]


def _is_stub(content: Any) -> bool:
  return isinstance(content, str) and content.startswith('{"_aged_tool_result"')


async def _completed(log: AgentSessionLog) -> dict[str, dict[str, Any]]:
  events, _ = await log.query(event_types={"tool_call_complete"})
  return {entry.event["tool_call_id"]: entry.event for entry in events}


def test_an_old_data_result_ages_but_instruction_and_small_results_never_do(tmp_path: Path) -> None:
  async def _case() -> None:
    provider, log, sink, runner = await _drive(tmp_path, list(SCRIPT), after_replies=3)
    final = _contents(provider.requests[-1])
    assert _is_stub(final["d1"])
    # Condition 5: skill / methodology bodies stay whole whatever their size.
    assert INSTRUCTION_RESULT_TOOLS == {"invoke_skill", "docs_fetch"}
    for instruction in ("s1", "f1"):
      assert len(final[instruction]) > 1000 and not _is_stub(final[instruction])
    # A result at or under the size bound never ages.
    assert not _is_stub(final["t1"])
    # A young result is whole: the first request after d1 carried it whole.
    assert not _is_stub(_contents(provider.requests[1])["d1"])

    # The durable log keeps the full result, and the stub only for the call that registered one.
    completed = await _completed(log)
    assert set(completed) >= {"d1", "s1", "f1", "t1"}
    assert "aging_stub" in completed["d1"]
    assert all("aging_stub" not in completed[name] for name in ("s1", "f1", "t1"))
    retained = completed["d1"]["final_tool_result_blocks"][0]["content"]
    assert json.loads(retained) == json.loads(DATA)
    assert runner._tool_result_aging_stubs == {"d1": completed["d1"]["aging_stub"]}
    assert final["d1"] == completed["d1"]["aging_stub"]
    assert len(final["d1"]) < len(retained)

    # The stub's ref resolves through the real tool_result_read to the retained content.
    stub = json.loads(final["d1"])
    handler = make_tool_result_read_handler(
      lambda: sink, caller=_reader, journal=lambda: log,
    )
    result, error = await handler({"spill_ref": stub["spill_ref"], "max_chars": 200_000})
    assert error is None and result is not None
    assert result["truncated"] is False
    assert json.loads(result["content"]) == json.loads(retained)

  asyncio.run(_case())


def test_requests_between_two_steps_share_one_aged_prefix(tmp_path: Path) -> None:
  async def _case() -> None:
    script: list[tuple[str, str] | None] = [("d1", "big_data")] + [(f"t{i}", "small") for i in range(8)] + [None]
    provider, *_ = await _drive(tmp_path, script, after_replies=2, step_replies=3)
    by_replies: dict[int, str] = {}
    young: list[int] = []
    for request in provider.requests:
      replies = sum(1 for message in request if message["role"] == "assistant")
      content = _contents(request).get("d1")
      if content is None:
        continue
      if _is_stub(content):
        by_replies[replies] = content
      else:
        young.append(replies)
    assert by_replies and young
    # d1 ages at a step, never between two, and then stays aged with identical bytes.
    assert max(young) < min(by_replies)
    assert min(by_replies) % 3 == 0
    assert len(set(by_replies.values())) == 1

  asyncio.run(_case())


def test_disabled_setting_leaves_every_request_unchanged(tmp_path: Path) -> None:
  async def _case() -> None:
    on, *_ = await _drive(tmp_path / "on", list(SCRIPT), after_replies=3)
    off, log, _, runner = await _drive(tmp_path / "off", list(SCRIPT), after_replies=None)
    assert not any(_is_stub(content) for request in off.requests for content in _contents(request).values())
    assert _is_stub(_contents(on.requests[-1])["d1"])
    assert runner._tool_result_aging_stubs == {}
    assert all("aging_stub" not in event for event in (await _completed(log)).values())

  asyncio.run(_case())


def test_a_later_turn_restores_stubs_from_the_log_so_earlier_results_age(tmp_path: Path) -> None:
  async def _case() -> None:
    _, log, _, _ = await _drive(
      tmp_path, [("d1", "big_data"), ("t1", "small"), None], after_replies=1,
    )
    stubs = await SessionContextBuilder(agent_session_log=log).aging_stubs()
    assert set(stubs) == {"d1"} and _is_stub(stubs["d1"])

    # A new runner on the same log replays the history: d1 is old from its first request.
    provider, *_ = await _drive(
      tmp_path,
      [("t2", "small"), ("t3", "small"), None],
      after_replies=1,
      log=log,
      context_builder=SessionContextBuilder(agent_session_log=log),
      prompt="again",
    )
    assert _contents(provider.requests[0])["d1"] == stubs["d1"]
    # With aging off the same replay carries the full result.
    off, *_ = await _drive(
      tmp_path,
      [None],
      after_replies=None,
      log=log,
      context_builder=SessionContextBuilder(agent_session_log=log),
      prompt="once more",
    )
    assert not _is_stub(_contents(off.requests[0])["d1"])

  asyncio.run(_case())
