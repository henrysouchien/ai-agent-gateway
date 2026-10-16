"""The Anthropic request tool prefix stays byte-stable while tools are added.

Drives the real runner (`_stream_turn`, `_execute_single_tool`) with the real
Anthropic request builder and a stream that records each request's params.
"""

import asyncio
import copy
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

PKG_DIR = Path(__file__).resolve().parents[1]
if str(PKG_DIR) not in sys.path:
  sys.path.insert(0, str(PKG_DIR))

from agent_gateway import AgentRunner, AgentSessionLog, EventLog, SessionContextBuilder  # noqa: E402
from agent_gateway.multi_user.billing import UsageTotals  # noqa: E402
from agent_gateway.providers import AnthropicProvider, CodexProvider, StreamEvent  # noqa: E402
from agent_gateway.session import GatewaySession  # noqa: E402
from agent_gateway.tool_dispatch_classification import settle_catalogless_tool_result  # noqa: E402
from agent_gateway.tool_policy_registry import PreparedToolCall  # noqa: E402
from gateway_test_support.capability_execution_test_support import (  # noqa: E402
  stub_bound_capability_execution,
)

MODEL = "claude-opus-5"
CONFIG = {"model": MODEL, "effort": "none", "auth_mode": "api_key", "api_key": "k"}


def _tool(name: str, **extra: Any) -> dict[str, Any]:
  return {
    "name": name,
    "description": f"{name} tool",
    "input_schema": {"type": "object", "properties": {}},
    **extra,
  }


class _Dispatcher:
  """Returns the scripted result for each tool name."""

  def __init__(self, results: dict[str, dict[str, Any]]) -> None:
    self.results = results

  def redact_prepared_tool_input(self, tool_name: str, prepared_call: Any) -> dict[str, Any]:
    return prepared_call.materialize_input()

  def settle_tool_result(self, tool_name, dispatch_entry, result, error, semantic_error=None, *, prepared_call):
    return settle_catalogless_tool_result(
      entry=dispatch_entry, result=result, error=error, semantic_error=semantic_error,
    )

  async def dispatch(self, tool_id, tool_name, tool_input, **_kwargs):
    return copy.deepcopy(self.results[tool_name]), None

  async def prepare_tool_call_async(self, tool_name, tool_input):
    _ = tool_name
    return PreparedToolCall(dict(tool_input))

  async def dispatch_prepared(self, tool_id, tool_name, prepared_call, **kwargs):
    return await self.dispatch(
      tool_id,
      tool_name,
      prepared_call.materialize_input(),
      **kwargs,
    )


class _Catalog:
  """The surface the runner's tool-definition getter reports right now."""

  def __init__(self, tools: list[dict[str, Any]]) -> None:
    self.tools = tools

  def __call__(self) -> list[dict[str, Any]]:
    return copy.deepcopy(self.tools)


def _session(session_id: str) -> GatewaySession:
  return GatewaySession(
    session_id=session_id, api_key_hash="h", created_at=0, expires_at=2**31,
    user_id="alice",
  )


def _runner(
  provider: Any,
  catalog: _Catalog,
  results: dict[str, dict[str, Any]],
  *,
  session: GatewaySession | None = None,
  excluded: set[str] | None = None,
  agent_session_log: Any = None,
) -> tuple[AgentRunner, list[dict[str, Any]]]:
  """A runner whose stream records every request's params."""

  requests: list[dict[str, Any]] = []

  async def stream(_client: Any, params: dict[str, Any]):
    requests.append(copy.deepcopy(params))
    yield StreamEvent(type="message_start")
    yield StreamEvent(type="text_delta", text="ok")
    yield StreamEvent(type="message_end", stop_reason="end_turn")

  provider.stream = stream
  execution = stub_bound_capability_execution(
    provider=provider, model=MODEL, effort="none", auth_config={"api_key": "k"},
  )
  runner = AgentRunner(
    event_log=EventLog(session_id=session.session_id if session else "deferred"),
    dispatcher=_Dispatcher(results),  # type: ignore[arg-type]
    session_id=session.session_id if session else "deferred",
    gateway_session=session,
    capability_execution=execution,
    get_tool_definitions=catalog,
    excluded_tools=excluded,
    agent_session_log=agent_session_log,
    user_id="alice",
    billing_mode="byok",
    rate_table_version="unknown",
  )
  return runner, requests


def _request(runner: AgentRunner, messages: list[dict[str, Any]], base_kwargs: dict[str, Any]) -> None:
  totals: UsageTotals = {
    "input_tokens": 0, "output_tokens": 0,
    "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0,
  }
  provider = runner._provider
  returned = asyncio.run(runner._stream_turn(
    client=object(), config=dict(CONFIG),
    model_info=provider.get_model_info(MODEL),
    system_prompt=None, current_messages=messages, base_kwargs=base_kwargs,
    max_tokens=4096, turn_count=1, turn_t0=time.time(), turn_t0_mono=time.monotonic(),
    system_chars=0, tools_chars=0, usage_totals=totals,
  ))
  assert isinstance(returned, tuple), returned


def _call(runner: AgentRunner, base_kwargs: dict[str, Any], tool_id: str, name: str) -> dict[str, Any]:
  live_entry, _name, _extra = asyncio.run(
    runner._execute_single_tool(tool_id, name, {}, base_kwargs)
  )
  return live_entry


def _turn_messages(history: list[dict[str, Any]], tool_id: str, name: str, entry: dict[str, Any]) -> None:
  history.append({
    "role": "assistant",
    "content": [{"type": "tool_use", "id": tool_id, "name": name, "input": {}}],
  })
  history.append({"role": "user", "content": [entry]})


def _bytes(value: Any) -> str:
  return json.dumps(value, sort_keys=True)


def _without_markers(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
  stripped = copy.deepcopy(messages)
  for message in stripped:
    if isinstance(message["content"], list):
      for block in message["content"]:
        block.pop("cache_control", None)
  return stripped


def _references(message: dict[str, Any]) -> list[str]:
  (block,) = [part for part in message["content"] if part["type"] == "tool_result"]
  return [
    part["tool_name"] for part in block["content"] if part["type"] == "tool_reference"
  ]


def test_tools_added_across_a_session_never_touch_the_cached_prefix() -> None:
  session = _session("deferred-session")
  catalog = _Catalog([_tool("read"), _tool("write", cache_control={"type": "ephemeral"}), _tool("skill_only")])
  results = {
    "load_tools": {"loaded_servers": ["m"], "_load_servers": ["m"]},
    "invoke_skill": {"ok": True, "_active_skill_allow": ["skill_only"]},
  }
  provider = AnthropicProvider()
  runner, requests = _runner(provider, catalog, results, session=session, excluded={"skill_only"})
  base_kwargs = {"tools": runner._filter_excluded_tool_definitions(runner._default_tool_definitions())}
  history: list[dict[str, Any]] = [{"role": "user", "content": "go"}]

  _request(runner, history, base_kwargs)
  first = requests[-1]
  # The anchor is the non-deferred tools. The unadvertised real definition
  # (skill_only) is carried as defer_loading from request 1 so Anthropic's
  # deferred-tools framing is established up front.
  assert [t["name"] for t in first["tools"] if not t.get("defer_loading")] == ["read", "write"]
  assert [(t["name"], t.get("defer_loading")) for t in first["tools"] if t.get("defer_loading")] == [
    ("skill_only", True),
  ]
  anchor_bytes = _bytes([t for t in first["tools"] if not t.get("defer_loading")])

  # load_tools adds two MCP tools; the tool_result of that call exposes them.
  catalog.tools += [_tool("mcp_b"), _tool("mcp_a")]
  entry = _call(runner, base_kwargs, "t1", "load_tools")
  assert entry["exposed_tools"] == ["mcp_a", "mcp_b"]
  _turn_messages(history, "t1", "load_tools", entry)
  _request(runner, history, base_kwargs)
  second = requests[-1]
  assert _bytes(second["tools"][:2]) == anchor_bytes
  assert [(t["name"], t.get("defer_loading")) for t in second["tools"][2:]] == [
    ("mcp_a", True), ("mcp_b", True), ("skill_only", True),
  ]
  assert all("cache_control" not in t for t in second["tools"][2:])
  assert _references(second["messages"][2]) == ["mcp_a", "mcp_b"]
  # The API rejects tool_reference mixed with other content in one tool_result:
  # the referencing tool_result holds only references, and the call's own
  # result follows it in the same user message, labelled by tool_use_id.
  tool_result, label, moved = second["messages"][2]["content"]
  assert {part["type"] for part in tool_result["content"]} == {"tool_reference"}
  assert label == {"type": "text", "text": "Result of tool call t1:"}
  assert moved["type"] == "text" and json.loads(moved["text"])["loaded_servers"] == ["m"]
  assert "exposed_tools" not in _bytes(second)

  # invoke_skill lifts the exclusion: the added tool is deferred and referenced.
  entry = _call(runner, base_kwargs, "t2", "invoke_skill")
  assert entry["exposed_tools"] == ["skill_only"]
  _turn_messages(history, "t2", "invoke_skill", entry)
  _request(runner, history, base_kwargs)
  third = requests[-1]
  assert _bytes(third["tools"][:2]) == anchor_bytes
  assert [t["name"] for t in third["tools"][2:]] == ["mcp_a", "mcp_b", "skill_only"]
  assert _references(third["messages"][4]) == ["skill_only"]
  assert _bytes(_without_markers(third["messages"][:3])) == _bytes(_without_markers(second["messages"]))

  # The run ends and the skill allow clears; the next turn is a new runner on
  # the same session, replaying the stored history.
  runner._active_skill_allow.clear()
  next_runner, next_requests = _runner(AnthropicProvider(), catalog, results, session=session, excluded={"skill_only"})
  next_kwargs = {"tools": next_runner._filter_excluded_tool_definitions(next_runner._default_tool_definitions())}
  assert "skill_only" not in {t["name"] for t in next_kwargs["tools"]}
  history.append({"role": "user", "content": "next question"})
  _request(next_runner, history, next_kwargs)
  fourth = next_requests[-1]
  assert _bytes(fourth["tools"][:2]) == anchor_bytes
  # The tool the history still references keeps its (deferred) definition.
  assert [t["name"] for t in fourth["tools"][2:]] == ["mcp_a", "mcp_b", "skill_only"]
  assert all(t["defer_loading"] is True for t in fourth["tools"][2:])
  # It is on the wire only for the reference: it is not advertised, so a fork
  # (whose ceiling reads the advertised names) cannot inherit it as authority.
  assert next_runner._last_request_advertised_tool_names == {"read", "write", "mcp_a", "mcp_b"}
  assert _bytes(_without_markers(fourth["messages"][:5])) == _bytes(_without_markers(third["messages"]))


def test_a_reference_is_dropped_when_its_tool_is_absent_from_the_request_tools() -> None:
  provider = AnthropicProvider()
  history = [
    {"role": "user", "content": "go"},
    {"role": "assistant", "content": [{"type": "tool_use", "id": "t1", "name": "load_tools", "input": {}}]},
    {"role": "user", "content": [{
      "type": "tool_result", "tool_use_id": "t1", "content": "{}", "exposed_tools": ["kept", "gone", "read"],
    }]},
  ]
  params = provider.build_request_params(
    model=MODEL, messages=history, system_prompt=None, max_tokens=4096,
    tools=[_tool("read"), _tool("kept")],
    tool_anchor=[_tool("read")],
  )
  assert [t["name"] for t in params["tools"]] == ["read", "kept"]
  assert _references(params["messages"][2]) == ["kept"]


def test_fork_child_sends_the_parents_exact_wire_tools_and_messages() -> None:
  session = _session("fork-parent")
  catalog = _Catalog([_tool("read", cache_control={"type": "ephemeral"})])
  results = {"load_tools": {"_load_servers": ["m"]}}
  runner, requests = _runner(AnthropicProvider(), catalog, results, session=session)
  base_kwargs = {"tools": runner._default_tool_definitions()}
  history: list[dict[str, Any]] = [{"role": "user", "content": "go"}]
  _request(runner, history, base_kwargs)
  catalog.tools += [_tool("mcp_a")]
  entry = _call(runner, base_kwargs, "t1", "load_tools")
  _turn_messages(history, "t1", "load_tools", entry)
  _request(runner, history, base_kwargs)
  parent_request = requests[-1]
  assert parent_request["tools"][-1] == {**_tool("mcp_a"), "defer_loading": True}

  child_catalog = _Catalog(runner._last_request_wire_tools)
  child, child_requests = _runner(AnthropicProvider(), child_catalog, {}, session=_session("fork-child"))
  setattr(child, "_fork_mode", True)
  setattr(child, "_fork_marker_position", runner._last_request_message_marker_position)
  _request(child, history, {"tools": child._default_tool_definitions()})
  assert _bytes(child_requests[-1]["tools"]) == _bytes(parent_request["tools"])
  assert _bytes(child_requests[-1]["messages"]) == _bytes(parent_request["messages"])


def test_other_providers_never_see_the_anchor_or_exposure() -> None:
  history = [
    {"role": "user", "content": "go"},
    {"role": "assistant", "content": [{"type": "tool_use", "id": "t1", "name": "load_tools", "input": {}}]},
    {"role": "user", "content": [{
      "type": "tool_result", "tool_use_id": "t1", "content": "{}", "exposed_tools": ["mcp_a"],
    }]},
  ]
  provider = CodexProvider()
  params = provider.build_request_params(
    model="gpt-6-sol", messages=history, system_prompt="sys",
    tools=[_tool("read"), _tool("mcp_a")], max_tokens=1024,
  )
  wire = json.dumps(params, default=str)
  assert "exposed_tools" not in wire
  assert "defer_loading" not in wire
  assert "tool_reference" not in wire
  assert [t["name"] for t in params["tools"]] == ["read", "mcp_a"]


def test_a_runner_not_on_anthropic_records_no_anchor_and_no_exposure() -> None:
  session = _session("codex-session")
  catalog = _Catalog([_tool("read")])
  runner, _ = _runner(AnthropicProvider(), catalog, {"load_tools": {"_load_servers": ["m"]}}, session=session)
  runner._provider = SimpleNamespace(name="codex")  # type: ignore[assignment]
  catalog.tools += [_tool("mcp_a")]
  base_kwargs = {"tools": runner._default_tool_definitions()}
  assert "exposed_tools" not in _call(runner, base_kwargs, "t1", "load_tools")
  assert session.request_tool_prefix.anchor is None
  assert session.request_tool_prefix.exposed == {}


@pytest.mark.parametrize("tools", [[], [_tool("only")]])
def test_first_request_is_unchanged(tools: list[dict[str, Any]]) -> None:
  runner, requests = _runner(AnthropicProvider(), _Catalog(tools), {})
  _request(runner, [{"role": "user", "content": "go"}], {"tools": runner._default_tool_definitions()})
  assert requests[-1]["tools"] == tools


def test_exposure_is_stored_on_the_durable_tool_result_the_history_is_rebuilt_from(tmp_path: Path) -> None:
  session_log = AgentSessionLog(path=tmp_path / "sessions" / "deferred.jsonl")
  session = _session("deferred-replay")
  catalog = _Catalog([_tool("read")])
  runner, _ = _runner(
    AnthropicProvider(), catalog, {"load_tools": {"_load_servers": ["m"]}},
    session=session, agent_session_log=session_log,
  )
  runner._runner_id = "writer-deferred"
  base_kwargs = {"tools": runner._default_tool_definitions()}
  _request(runner, [{"role": "user", "content": "go"}], base_kwargs)
  catalog.tools += [_tool("mcp_a")]
  _call(runner, base_kwargs, "t1", "load_tools")

  durable, _ = asyncio.run(session_log.query(order="asc"))
  (complete,) = [
    entry.event for entry in durable if entry.event.get("type") == "tool_call_complete"
  ]
  (block,) = complete["final_tool_result_blocks"]
  assert block["exposed_tools"] == ["mcp_a"]


def _conversation(
  session: GatewaySession,
  catalog: _Catalog,
  results: dict[str, dict[str, Any]] | None = None,
  *,
  excluded: set[str] | None = None,
  agent_session_log: Any = None,
) -> tuple[AgentRunner, list[dict[str, Any]], dict[str, Any]]:
  runner, requests = _runner(
    AnthropicProvider(), catalog, results or {}, session=session,
    excluded=excluded, agent_session_log=agent_session_log,
  )
  base_kwargs = {"tools": runner._filter_excluded_tool_definitions(runner._default_tool_definitions())}
  return runner, requests, base_kwargs


def _wire(request: dict[str, Any]) -> list[tuple[str, bool]]:
  return [(t["name"], t.get("defer_loading") is True) for t in request["tools"]]


def test_each_fresh_conversation_on_one_session_anchors_on_its_own_tools() -> None:
  """A pack runs skills one after another, each from fresh history, on one session."""

  session = _session("pack-session")
  first, first_requests, first_kwargs = _conversation(
    session, _Catalog([_tool("common"), _tool("skill1_tool")]),
  )
  _request(first, [{"role": "user", "content": "skill 1"}], first_kwargs)
  assert _wire(first_requests[-1]) == [("common", False), ("skill1_tool", False)]

  second, second_requests, second_kwargs = _conversation(
    session, _Catalog([_tool("common"), _tool("skill2_tool")]),
  )
  history: list[dict[str, Any]] = [{"role": "user", "content": "skill 2"}]
  _request(second, history, second_kwargs)
  assert _wire(second_requests[-1]) == [("common", False), ("skill2_tool", False)]

  history += [
    {"role": "assistant", "content": [{"type": "text", "text": "step"}]},
    {"role": "user", "content": "next"},
  ]
  _request(second, history, second_kwargs)
  assert _wire(second_requests[-1]) == [("common", False), ("skill2_tool", False)]


def test_compaction_that_drops_the_exposing_result_makes_loaded_tools_visible() -> None:
  session = _session("compaction-session")
  catalog = _Catalog([_tool("read")])
  results = {"load_tools": {"loaded_servers": ["m"], "_load_servers": ["m"]}}
  runner, requests, base_kwargs = _conversation(session, catalog, results)
  history: list[dict[str, Any]] = [{"role": "user", "content": "go"}]
  _request(runner, history, base_kwargs)
  catalog.tools += [_tool("mcp_a")]
  entry = _call(runner, base_kwargs, "t1", "load_tools")
  _turn_messages(history, "t1", "load_tools", entry)
  _request(runner, history, base_kwargs)
  assert _wire(requests[-1]) == [("read", False), ("mcp_a", True)]

  # Native compaction: the request carries history from the compaction block on.
  history += [
    {"role": "assistant", "content": [
      {"type": "compaction", "content": "summary of the load"},
      {"type": "text", "text": "continuing"},
    ]},
    {"role": "user", "content": "next"},
  ]
  _request(runner, history, base_kwargs)
  compacted = requests[-1]
  assert "tool_reference" not in _bytes(compacted["messages"])
  assert _wire(compacted) == [("read", False), ("mcp_a", False)]

  # Stable from here on: a later request keeps the same wire tools.
  history += [
    {"role": "assistant", "content": [{"type": "text", "text": "more"}]},
    {"role": "user", "content": "again"},
  ]
  _request(runner, history, base_kwargs)
  assert _bytes(requests[-1]["tools"]) == _bytes(compacted["tools"])


def test_a_tool_advertised_before_a_call_is_not_exposed_by_it() -> None:
  session = _session("diff-session")
  catalog = _Catalog([_tool("read"), _tool("late")])
  results = {"load_tools": {"loaded_servers": ["m"], "_load_servers": ["m"]}}
  runner, requests, base_kwargs = _conversation(session, catalog, results)
  history: list[dict[str, Any]] = [{"role": "user", "content": "go"}]
  _request(runner, history, base_kwargs)
  # `late` was advertised at the first request (so it is in the anchor); a load
  # that adds `fresh` exposes only `fresh`.
  catalog.tools += [_tool("fresh")]
  entry = _call(runner, base_kwargs, "t1", "load_tools")
  assert entry["exposed_tools"] == ["fresh"]
  _turn_messages(history, "t1", "load_tools", entry)
  _request(runner, history, base_kwargs)
  assert _wire(requests[-1]) == [("read", False), ("late", False), ("fresh", True)]


def test_exposure_round_trips_through_replay_into_identical_request_bytes(tmp_path: Path) -> None:
  session_log = AgentSessionLog(path=tmp_path / "sessions" / "replay.jsonl")
  session = _session("replay-session")
  catalog = _Catalog([_tool("read")])
  results = {"load_tools": {"loaded_servers": ["m"], "_load_servers": ["m"]}}
  runner, requests, base_kwargs = _conversation(
    session, catalog, results, agent_session_log=session_log,
  )
  runner._runner_id = "writer-replay"
  history: list[dict[str, Any]] = [{"role": "user", "content": "go"}]
  _request(runner, history, base_kwargs)
  tool_use = [{"type": "tool_use", "id": "t1", "name": "load_tools", "input": {}}]
  asyncio.run(session_log.append({"type": "user_message", "content": "go", "runner_id": "writer-replay"}))
  asyncio.run(session_log.append({"type": "assistant_message", "content_blocks": tool_use, "runner_id": "writer-replay"}))
  catalog.tools += [_tool("mcp_a")]
  entry = _call(runner, base_kwargs, "t1", "load_tools")
  _turn_messages(history, "t1", "load_tools", entry)
  history.append({"role": "assistant", "content": [{"type": "text", "text": "loaded"}]})
  history.append({"role": "user", "content": "next"})
  _request(runner, history, base_kwargs)
  live = requests[-1]

  rebuilt = asyncio.run(SessionContextBuilder(agent_session_log=session_log).build())
  (result_block,) = [
    block for message in rebuilt if isinstance(message["content"], list)
    for block in message["content"]
    if isinstance(block, dict) and block.get("type") == "tool_result"
  ]
  assert result_block["exposed_tools"] == ["mcp_a"]

  # Replaying the rebuilt tool_result in place of the live one sends the same bytes.
  replayed_history = [
    {**message, "content": [
      result_block if isinstance(block, dict) and block.get("type") == "tool_result" else block
      for block in message["content"]
    ]} if isinstance(message["content"], list) else message
    for message in history
  ]
  _request(runner, replayed_history, base_kwargs)
  assert _bytes(requests[-1]) == _bytes(live)
