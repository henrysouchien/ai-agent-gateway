"""Exercise real spill publication and durable append with a deterministic model."""

from __future__ import annotations

import json
from typing import Any

from agent_gateway import AgentRunner, AgentSessionLog, EventLog, ModelInfo, ModelProvider, ToolDispatcher
from agent_gateway.admission import ParentContext
from agent_gateway.mcp_client import McpClientManager
from agent_gateway.providers import StreamEvent
from agent_gateway.task_registry import RegistryOwner, TaskRegistry
from agent_gateway.tool_result_spill import SpillSink

from .capability_execution_test_support import stub_runner_capability_execution


class _SpillPublisherProvider(ModelProvider):
  name = "stub"

  def __init__(self, tool_use_id: str) -> None:
    self._tool_use_id = tool_use_id
    self._called_tool = False
    self.params_history: list[dict[str, Any]] = []
    self.last_spill_ref: str | None = None

  def has_active_credential(self, config: dict[str, Any]) -> bool:
    return True

  def create_client(self, config: dict[str, Any], *, timeout: float | None = None) -> Any:
    return object()

  async def close_client(self, client: Any, timeout: float = 2.0) -> None:
    pass

  def get_model_info(self, model: str) -> ModelInfo:
    return ModelInfo(id=model, provider=self.name)

  def build_request_params(
    self,
    *,
    model: str,
    messages: list[dict[str, Any]],
    system_prompt: str | list[tuple[str, bool]] | None,
    tools: list[dict[str, Any]],
    max_tokens: int,
    **kwargs: Any,
  ) -> dict[str, Any]:
    params = {
      "model": model, "messages": messages, "system_prompt": system_prompt,
      "tools": tools, "max_tokens": max_tokens, **kwargs,
    }
    self.params_history.append(params)
    for message in messages:
      blocks = message.get("content")
      if not isinstance(blocks, list):
        continue
      for block in blocks:
        if not isinstance(block, dict) or block.get("type") != "tool_result":
          continue
        content = block.get("content")
        if not isinstance(content, str):
          continue
        try:
          payload = json.loads(content)
        except json.JSONDecodeError:
          continue
        if isinstance(payload, dict) and isinstance(payload.get("spill_ref"), str):
          self.last_spill_ref = payload["spill_ref"]
    return params

  async def stream(self, client: Any, params: dict[str, Any]):
    yield StreamEvent(type="message_start", input_tokens=10)
    if not self._called_tool:
      self._called_tool = True
      yield StreamEvent(
        type="tool_use_end", tool_id=self._tool_use_id, tool_name="big_data",
        tool_input={}, raw_block={
          "type": "tool_use", "id": self._tool_use_id, "name": "big_data", "input": {},
        },
      )
      yield StreamEvent(type="message_end", stop_reason="tool_use")
    else:
      yield StreamEvent(type="text_delta", text="published")
      yield StreamEvent(type="text_end", raw_block={"type": "text", "text": "published"})
      yield StreamEvent(type="message_end", stop_reason="end_turn")


def _tool_definition() -> dict[str, Any]:
  return {
    "name": "big_data", "description": "",
    "input_schema": {"type": "object", "properties": {}},
  }


async def publish_custody_spill(
  sink: SpillSink,
  context: ParentContext,
  durable_log: AgentSessionLog,
  *,
  tool_use_id: str,
  payload: dict[str, Any],
) -> tuple[str, dict[str, Any], dict[str, Any], _SpillPublisherProvider]:
  """Run the actual publisher with a bound physical owner and fsynced journal."""
  assert context.job_id is not None and context.tenant is not None
  registry = TaskRegistry()
  registry.bind_owner(RegistryOwner(
    job_id=context.job_id,
    tenant=lambda: context.tenant,
    spill_sink=sink,
  ))

  async def _big_data(_tool_input: dict[str, Any], **_kwargs: Any):
    return payload, None

  event_log = EventLog()
  provider = _SpillPublisherProvider(tool_use_id)
  dispatcher = ToolDispatcher(
    mcp_client=McpClientManager(config_path=None),
    local_tool_handlers={"big_data": _big_data},
    event_log=event_log,
    session_id="session-alias",
    role="owner",
  )
  runner = AgentRunner(
    event_log=event_log,
    dispatcher=dispatcher,
    session_id="session-alias",
    capability_execution=stub_runner_capability_execution(
      provider=provider, auth_config={"api_key": "k"},
      model="stub-model", effort="none",
    ),
    get_tool_definitions=lambda: [_tool_definition()],
    user_id=context.tenant.storage_id,
    request_id=f"request:{tool_use_id}",
    billing_mode="byok",
    rate_table_version="unknown",
    task_registry=registry,
    agent_session_log=durable_log,
    code_execution_spill_dir_provider=sink,
  )
  await runner.run(messages=[{"role": "user", "content": "publish"}], system_prompt="x")
  events, _ = await durable_log.query(
    event_types={"tool_call_complete"}, tool_call_id=tool_use_id,
  )
  events = [
    entry for entry in events
    if entry.event["spill_publication"]["producer_job_id"] == context.job_id
  ]
  assert len(events) == 1
  live = next(
    entry.event for entry in event_log.entries
    if entry.event.get("type") == "tool_call_complete"
  )
  assert provider.last_spill_ref is not None
  return provider.last_spill_ref, events[0].event, live, provider
