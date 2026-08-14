from __future__ import annotations

import asyncio
from typing import Any

import pytest

from agent_gateway import AgentRunner, CostEstimate, EventLog, ModelInfo, ModelProvider
from agent_gateway.approval_policy import RunContext
from agent_gateway.runner import StreamTurnResult
from agent_gateway.run_identity import (
  MODEL_RUN_IDENTITY_MCP_TOOLS,
  RUN_IDENTITY_MCP_ENV,
  RUN_IDENTITY_MCP_TOOLS_ENV,
  RunIdentityCarrier,
  RunIdentityCarrierError,
  inject_run_identity_into_mcp_server_configs,
  model_run_identity_for_tool,
)
from agent_gateway.tool_dispatcher import ToolDispatcher
from tests.capability_execution_test_support import stub_bound_capability_execution


def _run(coro: Any) -> Any:
  return asyncio.run(coro)


class _McpClient:
  def __init__(self, tool_name: str, *, original_tool_name: str | None = None) -> None:
    self.tool_name = tool_name
    self.original_tool_name = original_tool_name
    self.calls: list[dict[str, Any]] = []

  def is_mcp_tool(self, name: str) -> bool:
    return name == self.tool_name

  def get_server_for_tool(self, name: str) -> str | None:
    return "portfolio-producers-mcp" if self.is_mcp_tool(name) else None

  def get_original_tool_name(self, name: str) -> str:
    if name == self.tool_name and self.original_tool_name is not None:
      return self.original_tool_name
    return name

  async def call_tool(
    self,
    name: str,
    tool_input: dict[str, Any],
    *,
    meta: dict[str, Any] | None = None,
  ) -> tuple[dict[str, Any], None]:
    self.calls.append({"name": name, "tool_input": tool_input, "meta": meta})
    return {"ok": True}, None

  def get_tool_definitions(self) -> list[dict[str, Any]]:
    return [
      {
        "name": self.tool_name,
        "description": self.tool_name,
        "input_schema": {"type": "object"},
      }
    ]


class _Provider(ModelProvider):
  name = "run-identity-test"

  def has_active_credential(self, _config: dict[str, Any]) -> bool:
    return True

  def create_client(
    self,
    _config: dict[str, Any],
    *,
    timeout: float | None = None,
  ) -> object:
    _ = timeout
    return object()

  async def close_client(self, _client: Any, timeout: float = 2.0) -> None:
    _ = timeout

  def get_model_info(self, model: str) -> ModelInfo:
    return ModelInfo(id=model, provider=self.name)

  def estimate_cost(
    self,
    model: str,
    uncached: int,
    output: int,
    *,
    cache_read_tokens: int = 0,
    cache_creation_tokens: int = 0,
  ) -> CostEstimate:
    _ = model, uncached, output, cache_read_tokens, cache_creation_tokens
    return CostEstimate()


def test_run_identity_carrier_routes_model_chains_and_rejects_author() -> None:
  carrier = RunIdentityCarrier("skill-run-1")

  assert {
    tool_name: model_run_identity_for_tool(tool_name, carrier)
    for tool_name in MODEL_RUN_IDENTITY_MCP_TOOLS
  } == {
    "build_model": "skill-run-1",
    "prepare_model_build": "skill-run-1",
  }
  assert model_run_identity_for_tool("fms_report_build_model", carrier) == "skill-run-1"
  assert model_run_identity_for_tool("filings_search", carrier) is None
  with pytest.raises(
    RunIdentityCarrierError,
    match="author_model_spec does not accept run identity",
  ):
    model_run_identity_for_tool("author_model_spec", carrier)


@pytest.mark.parametrize(
  "run_id",
  ["bad/run", "bad\\run", "bad\nrun", " leading", "trailing ", "x" * 129],
)
def test_run_identity_carrier_rejects_malformed_ids(run_id: str) -> None:
  with pytest.raises(
    RunIdentityCarrierError,
    match="canonical ASCII token",
  ) as raised:
    RunIdentityCarrier(run_id)

  assert raised.value.code == "run_identity_invalid"


def test_sdk_stdio_carrier_is_server_owned_and_copy_on_write() -> None:
  source = {
    "portfolio-producers-mcp": {
      "command": "python",
      "args": ["producer.py"],
      "env": {"EXISTING": "1"},
    },
    "research-corpus-mcp": {"command": "python", "args": ["corpus.py"]},
  }

  bound = inject_run_identity_into_mcp_server_configs(
    source,
    server_names=frozenset({"portfolio-producers-mcp"}),
    carrier=RunIdentityCarrier("skill-run-sdk"),
  )

  assert source["portfolio-producers-mcp"]["env"] == {"EXISTING": "1"}
  assert bound["portfolio-producers-mcp"]["env"] == {
    "EXISTING": "1",
    RUN_IDENTITY_MCP_ENV: "skill-run-sdk",
    RUN_IDENTITY_MCP_TOOLS_ENV: "build_model,prepare_model_build",
  }
  assert RUN_IDENTITY_MCP_ENV not in bound["research-corpus-mcp"].get("env", {})

  with pytest.raises(RunIdentityCarrierError, match="attempted to override"):
    inject_run_identity_into_mcp_server_configs(
      {
        "portfolio-producers-mcp": {
          "command": "python",
          "env": {RUN_IDENTITY_MCP_ENV: "attacker-run"},
        }
      },
      server_names=frozenset({"portfolio-producers-mcp"}),
      carrier=RunIdentityCarrier("skill-run-sdk"),
    )


@pytest.mark.parametrize("tool_name", ["prepare_model_build", "build_model"])
def test_custom_dispatcher_routes_same_run_identity_to_model_mcp_tools(
  tool_name: str,
) -> None:
  mcp = _McpClient(tool_name)
  run_context = RunContext(
    user_id="alice",
    request_id="request-1",
    session_id="session-1",
    run_id="skill-run-custom",
  )
  dispatcher = ToolDispatcher(
    mcp_client=mcp,  # type: ignore[arg-type]
    local_tool_handlers={},
    session_id="session-1",
    user_id="alice",
    risk_user_id=1,
    channel="web",
    role="owner",
    run_context=run_context,
    mcp_meta_inject_servers=frozenset({"portfolio-producers-mcp"}),
  )

  result, error = _run(
    dispatcher.dispatch(
      "call-1",
      tool_name,
      {},
      advertised_tool_names=frozenset({tool_name}),
      skill_run_id="skill-run-custom",
    )
  )

  assert error is None
  assert result == {"ok": True}
  assert run_context.run_id == "skill-run-custom"
  assert mcp.calls[0]["meta"]["skill_run_id"] == run_context.run_id


def test_custom_runner_preserves_request_snapshot_and_run_identity() -> None:
  async def case() -> None:
    mcp = _McpClient("build_model")
    event_log = EventLog()
    run_context = RunContext(
      user_id="alice",
      request_id="request-runner",
      session_id="session-runner",
      run_id="skill-run-custom",
    )
    dispatcher = ToolDispatcher(
      mcp_client=mcp,  # type: ignore[arg-type]
      local_tool_handlers={},
      event_log=event_log,
      session_id="session-runner",
      user_id="alice",
      risk_user_id=1,
      channel="web",
      role="owner",
      run_context=run_context,
      mcp_meta_inject_servers=frozenset({"portfolio-producers-mcp"}),
    )
    provider = _Provider()
    runner = AgentRunner(
      event_log=event_log,
      dispatcher=dispatcher,
      session_id="session-runner",
      capability_execution=stub_bound_capability_execution(
        provider=provider,  # type: ignore[arg-type]
        model="stub-model",
        effort="none",
        auth_config={"api_key": "test"},
      ),
      get_tool_definitions=mcp.get_tool_definitions,
      user_id="alice",
      billing_mode="byok",
      rate_table_version="unknown",
      skill_run_id="skill-run-custom",
    )
    turn_number = 0

    async def stream_turn(**_kwargs: Any):
      nonlocal turn_number
      turn_number += 1
      if turn_number == 1:
        return object(), StreamTurnResult(
          tool_uses=[("call-build", "build_model", {})],
          stop_reason="tool_use",
          content_blocks=[{
            "type": "tool_use",
            "id": "call-build",
            "name": "build_model",
            "input": {},
          }],
          advertised_tool_names=frozenset({"build_model"}),
        )
      return object(), StreamTurnResult(
        full_text="done",
        stop_reason="end_turn",
        content_blocks=[{"type": "text", "text": "done"}],
        advertised_tool_names=frozenset({"build_model"}),
      )

    runner._stream_turn = stream_turn  # type: ignore[method-assign]
    await runner.run(
      messages=[{"role": "user", "content": "build the model"}],
      max_turns=2,
    )

    assert len(mcp.calls) == 1
    assert mcp.calls[0]["meta"]["skill_run_id"] == run_context.run_id

  _run(case())


def test_custom_dispatcher_keeps_author_chain_identity_free() -> None:
  mcp = _McpClient("author_model_spec")
  dispatcher = ToolDispatcher(
    mcp_client=mcp,  # type: ignore[arg-type]
    local_tool_handlers={},
    session_id="session-1",
    user_id="alice",
    risk_user_id=1,
    channel="web",
    role="owner",
    run_context=RunContext(
      user_id="alice",
      request_id="request-1",
      session_id="session-1",
      run_id="skill-run-custom",
    ),
    mcp_meta_inject_servers=frozenset({"portfolio-producers-mcp"}),
  )

  result, error = _run(
    dispatcher.dispatch(
      "call-author",
      "author_model_spec",
      {},
      advertised_tool_names=frozenset({"author_model_spec"}),
      skill_run_id="skill-run-custom",
    )
  )

  assert error is None
  assert result == {"ok": True}
  assert "skill_run_id" not in mcp.calls[0]["meta"]


@pytest.mark.parametrize(
  ("skill_run_id", "context_run_id", "expected_code"),
  [
    (None, "skill-run-custom", "run_identity_required"),
    ("skill-run-custom", None, "run_identity_invalid"),
    ("skill-run-custom", "other-run", "run_identity_mismatch"),
  ],
)
def test_custom_dispatcher_fails_closed_on_missing_or_conflicting_identity(
  skill_run_id: str | None,
  context_run_id: str | None,
  expected_code: str,
) -> None:
  mcp = _McpClient("build_model")
  dispatcher = ToolDispatcher(
    mcp_client=mcp,  # type: ignore[arg-type]
    local_tool_handlers={},
    run_context=RunContext(
      user_id="alice",
      request_id="request-1",
      session_id="session-1",
      run_id=context_run_id,
    ),
    mcp_meta_inject_servers=frozenset({"portfolio-producers-mcp"}),
  )

  result, error = _run(
    dispatcher.dispatch(
      "call-build",
      "build_model",
      {},
      advertised_tool_names=frozenset({"build_model"}),
      skill_run_id=skill_run_id,
    )
  )

  assert result is None
  assert error is not None and error["code"] == expected_code
  assert mcp.calls == []


@pytest.mark.parametrize("original_tool_name", ["prepare_model_build", "build_model"])
def test_custom_dispatcher_normalizes_prefixed_model_tool_before_identity_check(
  original_tool_name: str,
) -> None:
  prefixed_name = f"mcp__portfolio-producers-mcp__{original_tool_name}"
  mcp = _McpClient(prefixed_name, original_tool_name=original_tool_name)
  dispatcher = ToolDispatcher(
    mcp_client=mcp,  # type: ignore[arg-type]
    local_tool_handlers={},
    run_context=RunContext(
      user_id="alice",
      request_id="request-1",
      session_id="session-1",
      run_id="skill-run-prefixed",
    ),
    mcp_meta_inject_servers=frozenset({"portfolio-producers-mcp"}),
  )

  result, error = _run(
    dispatcher.dispatch(
      "call-prefixed",
      prefixed_name,
      {},
      advertised_tool_names=frozenset({prefixed_name}),
      skill_run_id=None,
    )
  )

  assert result is None
  assert error is not None and error["code"] == "run_identity_required"
  assert mcp.calls == []


def test_custom_dispatcher_routes_prefixed_model_tool_identity_metadata() -> None:
  prefixed_name = "mcp__portfolio-producers-mcp__build_model"
  mcp = _McpClient(prefixed_name, original_tool_name="build_model")
  dispatcher = ToolDispatcher(
    mcp_client=mcp,  # type: ignore[arg-type]
    local_tool_handlers={},
    run_context=RunContext(
      user_id="alice",
      request_id="request-1",
      session_id="session-1",
      run_id="skill-run-prefixed",
    ),
    mcp_meta_inject_servers=frozenset({"portfolio-producers-mcp"}),
  )

  result, error = _run(
    dispatcher.dispatch(
      "call-prefixed",
      prefixed_name,
      {},
      advertised_tool_names=frozenset({prefixed_name}),
      skill_run_id="skill-run-prefixed",
    )
  )

  assert error is None
  assert result == {"ok": True}
  assert mcp.calls[0]["meta"]["skill_run_id"] == "skill-run-prefixed"


def test_custom_report_context_matches_runtime_run_identity() -> None:
  captured: dict[str, Any] = {}

  async def report_handler(
    _tool_input: dict[str, Any],
    *,
    tool_ctx: Any,
    **_kwargs: Any,
  ) -> tuple[dict[str, bool], None]:
    captured["skill_run_id"] = tool_ctx.skill_run_id
    captured["run_id"] = tool_ctx.run_id
    return {"recorded": True}, None

  dispatcher = ToolDispatcher(
    mcp_client=_McpClient("unused"),  # type: ignore[arg-type]
    local_tool_handlers={"fms_report_build_model": report_handler},
    run_context=RunContext(
      user_id="alice",
      request_id="request-1",
      session_id="session-1",
      run_id="skill-run-custom",
    ),
  )

  result, error = _run(
    dispatcher.dispatch(
      "call-report",
      "fms_report_build_model",
      {},
      skill_run_id="skill-run-custom",
    )
  )

  assert error is None
  assert result == {"recorded": True}
  assert captured == {
    "skill_run_id": "skill-run-custom",
    "run_id": "skill-run-custom",
  }
