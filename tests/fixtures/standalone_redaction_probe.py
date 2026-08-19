# ruff: noqa: E402

from __future__ import annotations

import asyncio
import io
import json
import logging
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
from typing import Any


installed = Path(sys.argv[1]).resolve()
state_root = Path(sys.argv[2]).resolve()
sys.path.insert(0, str(installed))

import agent_gateway
from agent_gateway import AgentRunner, EventLog, ToolDispatcher
from agent_gateway.agent_session_log import AgentSessionLog
from agent_gateway.capability_execution import BoundCapabilityExecution
from agent_gateway.model_registry import ModelRegistryEntry, ProductModelRegistry
from agent_gateway.providers import OpenAIProvider
from agent_workflow_contracts import CapabilityBind


SECRET = "sk-ant-api03-CODEX-WAVE0-CANARY-DO-NOT-USE-8f21d7"
TOOL_NAME = "data_historical_prices"
TOOL_DEFINITION = {
  "name": TOOL_NAME,
  "description": "Load historical prices",
  "input_schema": {
    "type": "object",
    "properties": {
      "symbol": {"type": "string"},
      "start_date": {"type": "string"},
      "end_date": {"type": "string"},
      "credential_note": {"type": "string"},
    },
    "required": ["symbol", "start_date", "end_date"],
    "additionalProperties": False,
  },
}
VALID_INPUT = {
  "symbol": "AAPL",
  "start_date": "2025-01-01",
  "end_date": "2025-01-31",
  "credential_note": SECRET,
}


class _EventStream:
  def __init__(self, events: list[dict[str, Any]]) -> None:
    self._events = events

  def __aiter__(self):
    self._iterator = iter(self._events)
    return self

  async def __anext__(self):
    try:
      return next(self._iterator)
    except StopIteration as exc:
      raise StopAsyncIteration from exc


class _Responses:
  def __init__(self, batches: list[list[dict[str, Any]]]) -> None:
    self._batches = batches
    self.requests: list[dict[str, Any]] = []

  async def create(self, **params: Any):
    self.requests.append(params)
    return _EventStream(self._batches[len(self.requests) - 1])


class _Provider(OpenAIProvider):
  def __init__(self, responses: _Responses) -> None:
    self._responses = responses

  def create_client(
    self,
    config: dict[str, Any],
    *,
    timeout: float | None = None,
  ) -> Any:
    _ = config, timeout
    return SimpleNamespace(responses=self._responses)

  async def close_client(self, client: Any, timeout: float = 2.0) -> None:
    _ = client, timeout


class _Mcp:
  def is_mcp_tool(self, _name: str) -> bool:
    return False

  def get_server_for_tool(self, _name: str) -> None:
    return None


def _execution(provider: OpenAIProvider) -> BoundCapabilityExecution:
  bind = CapabilityBind(
    schema_version="1.0",
    capability_id="session.driver",
    model_key="test.openai.gpt-5-6-terra",
    provider="openai",
    upstream_model="gpt-5.6-terra",
    adapter="test.openai",
    protocol_profile="test.reasoning",
    route="test.in_process",
    effort="low",
    credential_principal="service",
    credential_ref="test-service:openai",
    run_mode="interactive",
    registry_revision="test-redaction.1",
    policy_revision="test-redaction.1",
    selection_source="capability_default",
  )
  entry = ModelRegistryEntry(
    key=bind.model_key,
    label="OpenAI redaction probe",
    provider=bind.provider,
    upstream_model=bind.upstream_model,
    adapter=bind.adapter,
    protocol_profile=bind.protocol_profile,
    route=bind.route,
    lifecycle="active",
    capabilities={bind.capability_id: "internal"},
    supported_efforts=frozenset({bind.effort}),
    default_effort=bind.effort,
    features=frozenset({"tools", "streaming"}),
    reported_identities=frozenset({bind.upstream_model}),
  )
  registry = ProductModelRegistry(
    schema="product-model-registry/v1",
    revision=bind.registry_revision,
    models={entry.key: entry},
  )
  return BoundCapabilityExecution(
    bind=bind,
    registry=registry,
    adapter=provider,
    auth_config={
      "provider": "openai",
      "auth_mode": "api",
      "api_key": "sk-test",
      "max_tokens": 512,
    },
  )


def _function_call_events(
  call_id: str,
  tool_input: dict[str, Any],
) -> list[dict[str, Any]]:
  item = {
    "type": "function_call",
    "id": f"fc_{call_id}",
    "call_id": call_id,
    "name": TOOL_NAME,
    "arguments": json.dumps(tool_input),
  }
  return [
    {"type": "response.output_item.added", "item": dict(item)},
    {"type": "response.output_item.done", "item": dict(item)},
  ]


def _terminal_events() -> list[dict[str, Any]]:
  return [
    {
      "type": "response.output_item.added",
      "item": {
        "type": "message",
        "id": "msg_final",
        "status": "in_progress",
        "content": [],
      },
    },
    {
      "type": "response.content_part.added",
      "part": {"type": "output_text", "text": "", "annotations": []},
    },
    {"type": "response.output_text.delta", "delta": "done"},
    {
      "type": "response.output_item.done",
      "item": {
        "type": "message",
        "id": "msg_final",
        "status": "completed",
        "content": [
          {
            "type": "output_text",
            "text": "done",
            "annotations": [],
          }
        ],
      },
    },
    {
      "type": "response.completed",
      "response": {
        "status": "completed",
        "usage": {"input_tokens": 11, "output_tokens": 2},
      },
    },
  ]


def _tool_turn_events(tool_input: dict[str, Any]) -> list[dict[str, Any]]:
  return [
    *_function_call_events("call", tool_input),
    {
      "type": "response.completed",
      "response": {
        "status": "completed",
        "usage": {"input_tokens": 5, "output_tokens": 2},
      },
    },
  ]


async def _run_scenario(
  name: str,
  tool_input: dict[str, Any],
) -> dict[str, Any]:
  responses = _Responses([
    _tool_turn_events(tool_input),
    _terminal_events(),
  ])
  handler_inputs: list[dict[str, Any]] = []

  async def handler(value: dict[str, Any], **_kwargs: Any):
    handler_inputs.append(dict(value))
    return {"rows": 1}, None

  event_log = EventLog(session_id=name)
  dispatcher = ToolDispatcher(
    mcp_client=_Mcp(),
    local_tool_handlers={TOOL_NAME: handler},
    event_log=event_log,
    session_id=name,
    role="owner",
    get_tool_definitions=lambda: [TOOL_DEFINITION],
    local_tool_class_resolver=lambda _name: "read",
    local_catalog_action_resolver=lambda _name: None,
  )
  session_log = AgentSessionLog(path=state_root / f"{name}.jsonl")
  runner = AgentRunner(
    event_log=event_log,
    dispatcher=dispatcher,
    session_id=name,
    capability_execution=_execution(_Provider(responses)),
    get_tool_definitions=lambda: [TOOL_DEFINITION],
    user_id="alice",
    billing_mode="byok",
    rate_table_version="test",
    agent_session_log=session_log,
    emit_session_recap=False,
  )
  await runner.run([{"role": "user", "content": "load prices"}], max_turns=3)
  serialized_event_log = json.dumps([entry.event for entry in event_log.entries])
  serialized_history = json.dumps(responses.requests[1:])
  serialized_session = session_log.path.read_text()
  validation_errors = [
    entry.event
    for entry in event_log.entries
    if entry.event.get("type") == "tool_input_validation_failed"
  ]
  return {
    "handler_inputs": handler_inputs,
    "history": serialized_history,
    "surface": serialized_event_log + serialized_history + serialized_session,
    "validation_errors": validation_errors,
  }


async def _main() -> None:
  log_buffer = io.StringIO()
  log_handler = logging.StreamHandler(log_buffer)
  logging.getLogger("agent_gateway.runner").addHandler(log_handler)
  logging.getLogger("agent_gateway.runner").setLevel(logging.INFO)

  from agent_gateway import runner_tool_audit

  secret_loader, packaged_redactor = runner_tool_audit._tool_input_redactor()
  _ = secret_loader
  fallback = await _run_scenario("fallback", VALID_INPUT)

  agent_module = ModuleType("agent")
  agent_module.__path__ = []
  shared_module = ModuleType("agent.shared")
  shared_module.__path__ = []
  host_redaction_module = ModuleType("agent.shared.tool_redaction")
  host_redaction_module.get_audit_hmac_secret = lambda: b"broken-host-secret"

  def broken_redactor(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
    raise RuntimeError("broken selected host redactor")

  host_redaction_module.redact_tool_input = broken_redactor
  agent_module.shared = shared_module
  shared_module.tool_redaction = host_redaction_module
  sys.modules["agent"] = agent_module
  sys.modules["agent.shared"] = shared_module
  sys.modules["agent.shared.tool_redaction"] = host_redaction_module

  broken_valid = await _run_scenario(
    "broken-valid",
    VALID_INPUT,
  )
  broken_malformed = await _run_scenario(
    "broken-malformed",
    {"credential_note": SECRET},
  )
  all_logs = log_buffer.getvalue()
  print(json.dumps({
    "agent_gateway_file": str(Path(agent_gateway.__file__).resolve()),
    "fallback_handler_exact": fallback["handler_inputs"] == [VALID_INPUT],
    "fallback_history_has_fields": all(
      marker in fallback["history"]
      for marker in ("AAPL", "2025-01-01", "2025-01-31")
    ),
    "fallback_redactor_module": packaged_redactor.__module__,
    "fallback_surface_has_redaction": "<redacted-secret>" in fallback["surface"],
    "fallback_surface_has_secret": SECRET in fallback["surface"],
    "fallback_surface_has_tombstone": "_boundary_error" in fallback["surface"],
    "broken_malformed_handler_count": len(broken_malformed["handler_inputs"]),
    "broken_malformed_validation_errors": broken_malformed["validation_errors"],
    "broken_surface_has_secret": (
      SECRET in broken_valid["surface"]
      or SECRET in broken_malformed["surface"]
      or SECRET in all_logs
    ),
    "broken_valid_handler_exact": broken_valid["handler_inputs"] == [VALID_INPUT],
    "broken_valid_surface_has_tombstone": "_boundary_error" in broken_valid["surface"],
    "logs_have_secret": SECRET in all_logs,
  }, sort_keys=True))


asyncio.run(_main())
