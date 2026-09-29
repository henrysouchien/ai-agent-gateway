"""A child streaming a portable-compaction summary is live to its parent's guard."""

import asyncio
import sys
import time
from pathlib import Path
from typing import Any

import pytest

TESTS_DIR = Path(__file__).resolve().parent
if str(TESTS_DIR) not in sys.path:
  sys.path.insert(0, str(TESTS_DIR))

import agent_gateway.provider_summarize as provider_summarize  # noqa: E402
import agent_gateway.runner as gateway_runner  # noqa: E402
import agent_gateway.runner_sub_agents as runner_sub_agents  # noqa: E402
import agent_gateway.server_compaction as server_compaction  # noqa: E402
from agent_gateway.agent_session_log import AgentSessionLog  # noqa: E402
from agent_gateway.providers.base import StreamEvent  # noqa: E402
from agent_workflow_contracts import TaskResult  # noqa: E402
from gateway_test_support.capability_execution_test_support import (  # noqa: E402
  stub_bound_capability_execution,
)
from test_runner_sub_agents import _Provider, _parent, _spawn  # noqa: E402

_SUMMARY_CHUNK = "The child read the filings and kept every figure. "
_SUMMARY_WINDOW_S = 0.4


class _CompactingChildProvider(_Provider):
  """A child provider whose summary request outlasts the parent's activity gap."""

  def __init__(self, *, summary_streams_progress: bool) -> None:
    self._summary_streams_progress = summary_streams_progress

  def create_client(self, config: dict[str, Any], *, timeout: float | None = None) -> object:
    return object()

  async def close_client(self, client: Any, timeout: float = 2.0) -> None:
    return None

  def build_request_params(self, **kwargs: Any) -> dict[str, Any]:
    last = kwargs["messages"][-1]
    content = last.get("content") if isinstance(last, dict) else None
    return {
      "summary": content == server_compaction.DEFAULT_SERVER_COMPACTION_INSTRUCTIONS,
    }

  async def stream(self, client: Any, params: dict[str, Any]):
    if params["summary"]:
      deadline = time.monotonic() + _SUMMARY_WINDOW_S
      if self._summary_streams_progress:
        while time.monotonic() < deadline:
          await asyncio.sleep(0.01)
          yield StreamEvent(type="text_delta", text=_SUMMARY_CHUNK)
      else:
        # Keepalive pings only: a provider heartbeat is not progress.
        while time.monotonic() < deadline:
          await asyncio.sleep(0.01)
          yield StreamEvent(type="heartbeat")
        yield StreamEvent(type="text_delta", text=_SUMMARY_CHUNK * 8)
      yield StreamEvent(type="message_end", stop_reason="end_turn")
      return
    yield StreamEvent(type="text_delta", text="Done.")
    yield StreamEvent(type="text_end", raw_block={"type": "text", "text": "Done."})
    yield StreamEvent(type="message_end", stop_reason="end_turn")


@pytest.fixture
def _child_compacts_first_turn(monkeypatch: pytest.MonkeyPatch) -> None:
  monkeypatch.setattr(runner_sub_agents, "SUB_AGENT_ACTIVITY_GAP", 0.1)
  monkeypatch.setattr(runner_sub_agents, "STREAM_GUARD_POLL_INTERVAL", 0.01)
  monkeypatch.setattr(provider_summarize, "STREAM_PROGRESS_LOG_INTERVAL", 0.02)
  monkeypatch.setattr(gateway_runner, "STREAM_GUARD_POLL_INTERVAL", 0.02)
  monkeypatch.setattr(
    server_compaction,
    "should_portable_compact",
    lambda **_kwargs: (True, "forced"),
  )
  monkeypatch.setattr(
    server_compaction,
    "split_messages_for_compact",
    lambda messages, **_kwargs: ([dict(message) for message in messages], []),
  )


def _spawn_compacting_child(
  tmp_path: Path,
  *,
  summary_streams_progress: bool,
) -> tuple[Any, Any, list[object]]:
  parent = _parent(tmp_path, session_log=AgentSessionLog(tmp_path / "session.jsonl"))
  execution = stub_bound_capability_execution(
    provider=_CompactingChildProvider(
      summary_streams_progress=summary_streams_progress,
    ),
    model="child-model",
    effort="medium",
    capability_id="node.explore",
    credential_principal="user",
    auth_config={"api_key": "child-secret"},
  )
  observed: list[object] = []
  result, error = _spawn(
    parent,
    capability_execution=execution,
    on_sub_event=lambda event, _sid: observed.append(event.get("type")),
  )
  return result, error, observed


@pytest.mark.usefixtures("_child_compacts_first_turn")
def test_child_streaming_a_compaction_summary_outlasts_the_activity_gap(
  tmp_path: Path,
) -> None:
  result, error, observed = _spawn_compacting_child(
    tmp_path,
    summary_streams_progress=True,
  )

  assert error is None
  assert isinstance(result, TaskResult)
  assert result.execution.status == "succeeded"
  # The summary appends nothing to the child's log until it ends; its
  # heartbeats are what the parent saw before the compaction landed.
  compacted = observed.index("compaction")
  assert "heartbeat" in observed[:compacted]
  assert "error" not in observed


@pytest.mark.usefixtures("_child_compacts_first_turn")
def test_child_whose_compaction_summary_is_silent_is_settled_stalled(
  tmp_path: Path,
) -> None:
  result, error, observed = _spawn_compacting_child(
    tmp_path,
    summary_streams_progress=False,
  )

  assert error is None
  assert isinstance(result, TaskResult)
  assert result.execution.status == "interrupted"
  assert result.execution.terminal_reason is not None
  assert result.execution.terminal_reason.startswith(
    "stalled: Sub-agent stalled: no activity for"
  )
  assert "heartbeat" not in observed
  assert "compaction" not in observed
