from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Any, Never

import pytest


ROOT = Path(__file__).resolve().parents[3]
API_DIR = ROOT / "api"
if str(API_DIR) not in sys.path:
  sys.path.insert(0, str(API_DIR))

from agent.interactive.tool_dispatcher import AddinExecuteRequest, ExcelToolDispatcher  # noqa: E402
from agent_gateway.approval_route import DurableLocalApprovalRoute
from agent_gateway import ApprovalDecision, SessionStore, ToolDispatcher, ToolResult  # noqa: E402
from agent_gateway.code_execution import (  # noqa: E402
  CodeExecutionConfig,
  build_code_execution,
)
from agent_gateway.code_execution._backends import (  # noqa: E402
  DockerBackend,
  SubprocessBackend,
)
from agent_gateway.tool_dispatcher import InterceptDecision  # noqa: E402
from agent_gateway.tool_dispatch_classification import ToolResultSettlement  # noqa: E402
from agent_gateway.tool_policy_registry import PreparedToolCall  # noqa: E402
from agent_gateway.mcp_client import McpClientManager  # noqa: E402
from api.product_local_input_preparation_runtime import (  # noqa: E402
  bind_product_local_input_preparation_context,
)
from agent.shared.tool_policy_implementations import (  # noqa: E402
  product_redaction_context_factory,
)
from api.product_tool_registration import (  # noqa: E402
  product_tool_registration_composition,
)
from excel_mcp.relay import Channel, ChannelType  # noqa: E402
from agent_workflow_contracts.tool_registration import (  # noqa: E402
  ToolRegistrationDeclaration,
)




class _NoMcp(McpClientManager):
  def __init__(self) -> None:
    super().__init__(config_path=None)

  def is_mcp_tool(self, name: str) -> bool:
    _ = name
    return False

  def get_server_for_tool(self, name: str) -> None:
    _ = name
    return None

  async def call_tool(self, *_args: object, **_kwargs: object) -> Never:
    raise AssertionError("registered code_execute must stay on its local route")


def _session():
  return SessionStore(ttl=3600).create_session(
    api_key_hash="hash",
    user_id="alice",
  )

async def _unused_execute_addin(_request: AddinExecuteRequest) -> ToolResult:
  raise AssertionError("registered local code must not use the add-in relay")


def _dispatcher(
  bundle: Any,
  *,
  request_approval: Any = None,
  interceptors: list[Any] | None = None,
) -> ToolDispatcher:
  composition = product_tool_registration_composition()
  return ToolDispatcher(
    mcp_client=_NoMcp(),
    local_tool_handlers=bundle.handlers,
    role="owner",
    needs_approval=bundle.needs_approval,
    request_approval=request_approval,
    approved_tool_types=set(),
    interceptors=interceptors,
    get_tool_definitions=lambda: bundle.tool_definitions,
    tool_registration_catalog=composition.catalog,
    tool_policy_implementations=composition.policy_implementations,
    input_preparation_context_factory=(
      bind_product_local_input_preparation_context(bundle)
    ),
    redaction_context_factory=product_redaction_context_factory,
  )


def _success(stdout: str = "ok\n") -> dict[str, Any]:
  return {
    "stdout": stdout,
    "stderr": "",
    "return_code": 0,
    "images": [],
    "timed_out": False,
    "duration_ms": 1,
    "truncated": False,
  }


def test_registered_auto_backend_is_selected_before_approval_and_shared_with_handler(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  available_calls = 0

  def available(_self: SubprocessBackend) -> bool:
    nonlocal available_calls
    available_calls += 1
    return True

  async def execute(
    _self: SubprocessBackend,
    _code: str,
    _work_dir: str,
    **_kwargs: Any,
  ) -> dict[str, Any]:
    return _success()

  monkeypatch.setattr(SubprocessBackend, "available", available)
  monkeypatch.setattr(SubprocessBackend, "execute", execute)
  bundle = build_code_execution(
    _session(),
    CodeExecutionConfig(register_docker=False),
  )
  approval_qualifiers: list[str] = []
  handler_qualifiers: list[str] = []
  original_handler = bundle.handlers["code_execute"]

  async def handler(tool_input: dict[str, Any], **kwargs: Any):
    handler_qualifiers.append(kwargs["tool_ctx"].resolved_qualifier)
    return await original_handler(tool_input, **kwargs)

  bundle.handlers["code_execute"] = handler

  async def approve(request: Any) -> ApprovalDecision:
    approval_qualifiers.append(request.resolved_qualifier)
    return ApprovalDecision(approved=True)

  dispatcher = _dispatcher(bundle, request_approval=approve)
  prepared = dispatcher.prepare_tool_call(
    "code_execute",
    {"code": "print('ok')", "host": "auto"},
  )

  assert prepared.exact_backend == "subprocess"
  assert available_calls == 1
  assert dispatcher.requires_approval_prepared("code_execute", prepared) is True
  assert available_calls == 1

  result, error = asyncio.run(
    dispatcher.dispatch_prepared("call-1", "code_execute", prepared)
  )

  assert error is None
  assert result is not None and result["stdout"] == "ok\n"
  assert approval_qualifiers == ["subprocess"]
  assert handler_qualifiers == ["subprocess"]
  assert available_calls == 2


def test_same_registered_prepared_call_retries_without_backend_reselection(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  docker_available_calls = 0

  def docker_available(_self: DockerBackend) -> bool:
    nonlocal docker_available_calls
    docker_available_calls += 1
    return True

  def subprocess_available(_self: SubprocessBackend) -> bool:
    raise AssertionError("a prepared docker call must not fall back to subprocess")

  async def docker_execute(
    _self: DockerBackend,
    _code: str,
    _work_dir: str,
    **_kwargs: Any,
  ) -> dict[str, Any]:
    return _success("settled\n")

  monkeypatch.setattr(DockerBackend, "available", docker_available)
  monkeypatch.setattr(DockerBackend, "execute", docker_execute)
  monkeypatch.setattr(SubprocessBackend, "available", subprocess_available)
  bundle = build_code_execution(_session())
  preparation_calls = 0
  prepare_call = bundle.prepare_call

  def counted_prepare(raw_input: dict[str, Any]) -> PreparedToolCall:
    nonlocal preparation_calls
    preparation_calls += 1
    return prepare_call(raw_input)

  bundle.prepare_call = counted_prepare
  handler_qualifiers: list[str] = []
  original_handler = bundle.handlers["code_execute"]

  async def fail_once(tool_input: dict[str, Any], **kwargs: Any):
    handler_qualifiers.append(kwargs["tool_ctx"].resolved_qualifier)
    if len(handler_qualifiers) == 1:
      return None, {"code": "internal_error", "message": "connection reset"}
    return await original_handler(tool_input, **kwargs)

  bundle.handlers["code_execute"] = fail_once
  dispatcher = _dispatcher(bundle)
  raw_input = {"code": "print('settled')", "host": "auto"}
  prepared = dispatcher.prepare_tool_call("code_execute", raw_input)
  raw_input["host"] = "subprocess"

  first = asyncio.run(
    dispatcher.dispatch_prepared("call-1", "code_execute", prepared)
  )
  second = asyncio.run(
    dispatcher.dispatch_prepared("call-1", "code_execute", prepared)
  )

  assert first == (
    None,
    {"code": "internal_error", "message": "connection reset"},
  )
  assert second[1] is None
  assert second[0] is not None and second[0]["stdout"] == "settled\n"
  assert prepared.materialize_input()["host"] == "auto"
  assert prepared.exact_backend == "docker"
  assert preparation_calls == 1
  assert handler_qualifiers == ["docker", "docker"]
  assert docker_available_calls == 2


@pytest.mark.parametrize(
  ("tool_input", "message"),
  [
    ({"code": "print(1)", "host": "missing"}, "Unknown host: 'missing'"),
    ({"code": 3}, "code must be a string"),
    ({"code": "print(1)", "background": "yes"}, "background must be a boolean"),
    ({"code": "print(1)", "timeout_ms": "1000"}, "timeout_ms must be an integer"),
  ],
)
def test_registered_invalid_input_stops_before_approval_and_handler(
  tool_input: dict[str, Any],
  message: str,
) -> None:
  bundle = build_code_execution(
    _session(),
    CodeExecutionConfig(register_docker=False),
  )
  approval_calls = 0
  handler_calls = 0

  async def approve(_request: Any) -> ApprovalDecision:
    nonlocal approval_calls
    approval_calls += 1
    return ApprovalDecision(approved=True)

  async def handler(_tool_input: dict[str, Any], **_kwargs: Any):
    nonlocal handler_calls
    handler_calls += 1
    return _success(), None

  bundle.handlers["code_execute"] = handler
  dispatcher = _dispatcher(bundle, request_approval=approve)

  result, error = asyncio.run(
    dispatcher.dispatch("invalid", "code_execute", tool_input)
  )

  assert result is None
  assert error == {"code": "invalid_input", "message": message}
  assert approval_calls == 0
  assert handler_calls == 0


def test_prepared_auto_backend_becoming_unavailable_does_not_fall_back(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  docker_states = iter((True, False))
  subprocess_calls = 0

  def docker_available(_self: DockerBackend) -> bool:
    return next(docker_states)

  def subprocess_available(_self: SubprocessBackend) -> bool:
    nonlocal subprocess_calls
    subprocess_calls += 1
    return True

  async def execute(*_args: Any, **_kwargs: Any):
    raise AssertionError("an unavailable prepared backend must not execute")

  monkeypatch.setattr(DockerBackend, "available", docker_available)
  monkeypatch.setattr(DockerBackend, "execute", execute)
  monkeypatch.setattr(SubprocessBackend, "available", subprocess_available)
  bundle = build_code_execution(_session())
  dispatcher = _dispatcher(bundle)
  prepared = dispatcher.prepare_tool_call(
    "code_execute",
    {"code": "print(1)", "host": "auto"},
  )

  result, error = asyncio.run(
    dispatcher.dispatch_prepared("call-1", "code_execute", prepared)
  )

  assert prepared.exact_backend == "docker"
  assert result is None
  assert error == {
    "code": "backend_unavailable",
    "message": "Backend 'docker' unavailable",
  }
  assert subprocess_calls == 0


def test_interceptor_cannot_mutate_registered_prepared_code_input(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  monkeypatch.setattr(SubprocessBackend, "available", lambda _self: True)
  bundle = build_code_execution(
    _session(),
    CodeExecutionConfig(register_docker=False),
  )
  handler_calls = 0
  approval_calls = 0

  async def mutate(context: Any) -> InterceptDecision:
    context.tool_input["code"] = "print('changed')"
    return InterceptDecision("allow")

  async def handler(_tool_input: dict[str, Any], **_kwargs: Any):
    nonlocal handler_calls
    handler_calls += 1
    return _success(), None

  async def approve(_request: Any) -> ApprovalDecision:
    nonlocal approval_calls
    approval_calls += 1
    return ApprovalDecision(approved=True)

  bundle.handlers["code_execute"] = handler
  dispatcher = _dispatcher(
    bundle,
    request_approval=approve,
    interceptors=[mutate],
  )
  prepared = dispatcher.prepare_tool_call(
    "code_execute",
    {"code": "print('original')", "host": "subprocess"},
  )

  result, error = asyncio.run(
    dispatcher.dispatch_prepared("call-1", "code_execute", prepared)
  )

  assert result is None
  assert error is not None and error["code"] == "tool_input_preparation_failed"
  assert prepared.materialize_input()["code"] == "print('original')"
  assert approval_calls == 0
  assert handler_calls == 0


def test_approval_cannot_replace_registered_prepared_code_input(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  monkeypatch.setattr(SubprocessBackend, "available", lambda _self: True)
  bundle = build_code_execution(
    _session(),
    CodeExecutionConfig(register_docker=False),
  )
  handler_calls = 0

  async def handler(_tool_input: dict[str, Any], **_kwargs: Any):
    nonlocal handler_calls
    handler_calls += 1
    return _success(), None

  bundle.handlers["code_execute"] = handler
  dispatcher = _dispatcher(bundle)

  async def modified_approval(**kwargs: Any) -> dict[str, Any]:
    return {
      "approved": True,
      "allow_tool_type": False,
      "tool_input": {**kwargs["tool_input"], "code": "print('changed')"},
    }

  dispatcher._approval_route = DurableLocalApprovalRoute(
    object(),
    object(),
    SessionStore(ttl=3600).create_session(
      api_key_hash="hash",
      user_id="alice",
      role="owner",
    ),
  )
  monkeypatch.setattr(
    dispatcher,
    "_run_approval_lifecycle",
    modified_approval,
  )
  prepared = dispatcher.prepare_tool_call(
    "code_execute",
    {"code": "print('original')", "host": "subprocess"},
  )

  result, error = asyncio.run(
    dispatcher.dispatch_prepared("call-1", "code_execute", prepared)
  )

  assert result is None
  assert error is not None and error["code"] == "tool_input_preparation_failed"
  assert prepared.materialize_input()["code"] == "print('original')"
  assert handler_calls == 0


class _PreparedBaseSpy:
  def __init__(self) -> None:
    self._local = {"code_execute": object()}
    self._request_approval = object()
    self.prepared = PreparedToolCall(
      {"code": "print(1)", "host": "auto"},
      "docker",
    )
    self.approval_calls: list[PreparedToolCall] = []
    self.dispatch_calls: list[PreparedToolCall] = []
    self.outcome_calls: list[tuple[str, Any, Any, Any, Any]] = []
    self.registered_outcome_calls: list[tuple[str, Any, Any, Any]] = []

  def registered_addin_declaration(
    self,
    tool_name: str,
  ) -> ToolRegistrationDeclaration | None:
    return None

  def prepare_tool_call(
    self,
    _tool_name: str,
    _tool_input: dict[str, Any],
  ) -> PreparedToolCall:
    return self.prepared

  def requires_approval_prepared(
    self,
    _tool_name: str,
    prepared_call: PreparedToolCall,
  ) -> bool:
    self.approval_calls.append(prepared_call)
    return True

  def _needs_approval(
    self,
    _tool_name: str,
    tool_input: dict[str, Any],
    exact_backend: str,
  ) -> bool:
    assert tool_input == self.prepared.materialize_input()
    assert exact_backend == self.prepared.exact_backend
    self.approval_calls.append(self.prepared)
    return True

  async def dispatch_prepared(
    self,
    _tool_call_id: str,
    _tool_name: str,
    prepared_call: PreparedToolCall,
    **_kwargs: Any,
  ):
    self.dispatch_calls.append(prepared_call)
    return {"ok": True}, None

  @staticmethod
  def ensure_gateway_local_tool_handler(_tool_name: str) -> bool:
    return True

  def settle_tool_result(
    self,
    tool_name: str,
    dispatch_entry: Any,
    result: Any,
    error: Any,
    semantic_error: Any = None,
    *,
    prepared_call: PreparedToolCall,
  ) -> ToolResultSettlement:
    assert prepared_call is self.prepared
    self.outcome_calls.append(
      (tool_name, dispatch_entry, result, error, semantic_error)
    )
    return ToolResultSettlement("ok")

  @staticmethod
  def registered_outcomes_configured() -> bool:
    return True

  @staticmethod
  def uses_registered_tool_catalog() -> bool:
    return True

  def settle_registered_addin_outcome(
    self,
    tool_name: str,
    result: Any,
    error: Any,
    semantic_error: Any = None,
  ) -> str:
    self.registered_outcome_calls.append(
      (tool_name, result, error, semantic_error)
    )
    return "error_semantic"


class _NoChannels:
  @staticmethod
  def get_active_channels() -> list[Channel]:
    return []

  @staticmethod
  def get_channel_for_tool(_tool_name: str) -> Channel | None:
    return None


def test_excel_dispatcher_forwards_exact_prepared_call_and_classification() -> None:


  channel = Channel(
    channel_id="excel-collision",
    channel_type=ChannelType.EXCEL,
    tool_names={"code_execute"},
  )

  class _ExcelCollisionChannels(_NoChannels):
    @staticmethod
    def get_channel_for_tool(tool_name: str):
      return channel if tool_name == "code_execute" else None

  base = _PreparedBaseSpy()
  dispatcher = ExcelToolDispatcher(
    base=base,  # type: ignore[arg-type]
    channel_registry=_ExcelCollisionChannels(),  # type: ignore[arg-type]
    execute_addin=_unused_execute_addin,
  )

  prepared = dispatcher.prepare_tool_call("code_execute", {"code": "ignored"})
  requires_approval = dispatcher.requires_approval_prepared(
    "code_execute",
    prepared,
  )
  result, error = asyncio.run(
    dispatcher.dispatch_prepared("call-1", "code_execute", prepared)
  )
  settlement = dispatcher.settle_tool_result(
    "code_execute",
    None,
    result,
    error,
    prepared_call=prepared,
  )

  assert requires_approval is True
  assert error is None
  assert result == {"ok": True}
  assert settlement.outcome == "ok"
  assert prepared is base.prepared
  assert base.approval_calls == [prepared]
  assert base.dispatch_calls == [prepared]
  assert base.outcome_calls == [
    ("code_execute", None, {"ok": True}, None, None)
  ]


def test_excel_dispatcher_activates_exact_registered_addin_outcome() -> None:

  channel = Channel(
    channel_id="excel-outcome",
    channel_type=ChannelType.EXCEL,
    tool_names={"read_cells"},
  )

  class _ExcelChannels(_NoChannels):
    @staticmethod
    def get_channel_for_tool(tool_name: str):
      return channel if tool_name == "read_cells" else None

  base = _PreparedBaseSpy()
  dispatcher = ExcelToolDispatcher(
    base=base,  # type: ignore[arg-type]
    channel_registry=_ExcelChannels(),  # type: ignore[arg-type]
    execute_addin=_unused_execute_addin,
  )

  assert dispatcher.settle_tool_result(
    "read_cells",
    None,
    {"status": "error", "error": {"code": "bad_range"}},
    None,
    prepared_call=PreparedToolCall({"range": "A1"}),
  ).outcome == "error_semantic"
  assert base.outcome_calls == []
  assert base.registered_outcome_calls == [(
    "read_cells",
    {"status": "error", "error": {"code": "bad_range"}},
    None,
    None,
  )]


def test_excel_dispatcher_forwards_mcp_outcome_to_base_owner() -> None:

  channel = Channel(
    channel_id="mcp-outcome",
    channel_type=ChannelType.MCP_EXTERNAL,
    tool_names={"provider__read"},
  )

  class _McpChannels(_NoChannels):
    @staticmethod
    def get_channel_for_tool(tool_name: str):
      return channel if tool_name == "provider__read" else None

  base = _PreparedBaseSpy()
  dispatcher = ExcelToolDispatcher(
    base=base,  # type: ignore[arg-type]
    channel_registry=_McpChannels(),  # type: ignore[arg-type]
    execute_addin=_unused_execute_addin,
  )
  result = {"status": "success"}

  assert dispatcher.settle_tool_result(
    "provider__read",
    None,
    result,
    None,
    prepared_call=base.prepared,
  ).outcome == "ok"
  assert base.outcome_calls == [
    ("provider__read", None, result, None, None)
  ]
