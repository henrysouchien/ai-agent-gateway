from __future__ import annotations

import asyncio
import json
from collections.abc import Callable, Mapping
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from gateway_test_support.host_policy import owner_session_host_policy

from agent_gateway.approval_route import DurableLocalApprovalRoute
from agent_gateway.approval_store import SQLiteApprovalStore
from agent_gateway.session import SessionStore
from agent_gateway import ApprovalDecision, ToolDispatcher
from agent_gateway.mcp_client import McpClientManager, RegisteredMcpDirectToolCall
from agent_gateway.single_user_policy import SingleUserApprovalPolicy
from agent_gateway.tool_definition import LiveToolRouteBinding, OriginatedToolDefinition
from agent_gateway.tool_dispatcher_helpers import ToolResult, TransportApprovalRequest
from agent_gateway.tool_policy_registry import (
  PlanDecision,
  PreparedToolCall,
  RedactionResult,
  SourceIdentityResult,
  ToolPolicyImplementation,
  ToolPolicyImplementationRegistry,
  reject_policy_parameters,
)
from agent_gateway.tool_registration import (
  RegisteredMcpToolDescriptor,
  compile_registered_mcp_tool_descriptors,
)
from agent_gateway.runner_tool_audit import redact_tool_input_for_event
from agent_gateway.tool_dispatcher_approval_lifecycle import (
  hash_approval_arguments,
)
from agent_workflow_contracts.tool_registration import (
  PolicyKind,
  RegisteredToolIdentity,
  RegisteredToolServerDescriptor,
  ToolApprovalPolicy,
  ToolIntrinsicSemantics,
  ToolRegistrationCatalog,
  ToolRegistrationDeclaration,
  ToolRouteKind,
  VersionedPolicyRef,
)


def _ref(kind: PolicyKind, policy_id: str | None = None) -> VersionedPolicyRef:
  return VersionedPolicyRef(
    kind=kind,
    policy_id=policy_id or kind,
    version="v1",
  )


def _entry(
  kind: PolicyKind,
  implementation: Any,
  *,
  policy_id: str | None = None,
) -> ToolPolicyImplementation:
  return ToolPolicyImplementation(
    kind,
    policy_id or kind,
    "v1",
    reject_policy_parameters,
    implementation,
  )


def _declaration(
  approval: ToolApprovalPolicy,
  *,
  route_kind: ToolRouteKind = "local_handler",
  name: str = "registered_write",
  server_id: str | None = None,
  outcome_policy: VersionedPolicyRef | None = None,
  planning_policy: VersionedPolicyRef | None = None,
) -> ToolRegistrationDeclaration:
  return ToolRegistrationDeclaration(
    RegisteredToolIdentity(
      route_kind=route_kind,
      logical_server_id=server_id,
      logical_name=name,
    ),
    ToolIntrinsicSemantics(
      effect="external_write",
      idempotent=False,
      semantic_capability="test.write/v1",
      approval=approval,
      audience="ordinary",
      redaction_policy=_ref("redaction"),
      planning_policy=planning_policy or _ref("planning", "none"),
      input_preparation_policy=_ref("input_preparation"),
      outcome_policy=outcome_policy or _ref("outcome"),
      source_identity_policy=_ref("source_identity"),
    ),
  )


def _registry(
  *,
  predicate: Any = lambda _ref, _call: True,
  cache_key: Any = lambda _ref, _call: "safe-key",
  cache_key_policy_id: str | None = None,
  planning_policy_id: str = "none",
  redaction: Any = lambda _ref, call: RedactionResult(call.prepared_input),
  outcome: Any = lambda _ref, _call: "ok",
  extra_outcomes: tuple[tuple[str, Any], ...] = (),
) -> ToolPolicyImplementationRegistry:
  return ToolPolicyImplementationRegistry((
    _entry("approval_predicate", predicate),
    _entry(
      "approval_cache_key",
      cache_key,
      policy_id=cache_key_policy_id,
    ),
    _entry("redaction", redaction),
    _entry(
      "planning",
      lambda _ref, _call: PlanDecision("none"),
      policy_id=planning_policy_id,
    ),
    _entry(
      "input_preparation",
      lambda _ref, call: PreparedToolCall(call.raw_input),
    ),
    _entry("outcome", outcome),
    *(
      _entry("outcome", implementation, policy_id=policy_id)
      for policy_id, implementation in extra_outcomes
    ),
    _entry(
      "source_identity",
      lambda _ref, _call: SourceIdentityResult(()),
    ),
    _entry(
      "session_injection",
      lambda _ref, _call: (_ for _ in ()).throw(
        AssertionError("local route cannot inject an MCP session")
      ),
    ),
  ))


class _NoMcp(McpClientManager):
  def __init__(self) -> None:
    super().__init__(config_path=None)

  async def call_tool(
    self,
    name: str,
    tool_input: object,
    meta: object | None = None,
    abort_event: asyncio.Event | None = None,
    gateway_session: object | None = None,
    allow_uncertain_replay: bool = True,
    trusted_dispatch_scope: object | None = None,
  ) -> ToolResult:
    _ = (
      name,
      tool_input,
      meta,
      abort_event,
      gateway_session,
      allow_uncertain_replay,
      trusted_dispatch_scope,
    )
    raise AssertionError("local registered route must not call MCP")

def _registered_descriptor(
  exposed_name: str,
  declaration: ToolRegistrationDeclaration,
  server: RegisteredToolServerDescriptor,
) -> RegisteredMcpToolDescriptor:
  logical_server_id = declaration.identity.logical_server_id
  assert logical_server_id is not None
  logical_name = declaration.identity.logical_name
  binding = LiveToolRouteBinding(
    originated_definition=OriginatedToolDefinition(
      definition={
        "name": exposed_name,
        "input_schema": {"type": "object"},
      },
      origin="mcp",
      server_id=logical_server_id,
    ),
    route_kind="physical",
    logical_name=logical_name,
    transport_server_id=server.transport_server_id,
    provider_original_name=logical_name,
    provider_id="provider",
  )
  descriptor, = compile_registered_mcp_tool_descriptors(
    ToolRegistrationCatalog((declaration,), (server,)),
    (binding,),
  )
  assert descriptor.exposed_name == exposed_name
  return descriptor


async def _local_handler(tool_input: dict[str, Any], **_kwargs: Any):
  return {"received": tool_input}, None


def _local_dispatcher(
  declaration: ToolRegistrationDeclaration,
  registry: ToolPolicyImplementationRegistry,
  *,
  needs_approval: Any,
  request_approval: Any = None,
  approved: set[str] | None = None,
  context_factory: Any = lambda _declaration, _prepared_call: None,
  redaction_context_factory: Any = lambda _declaration: None,
  overlay: Any = None,
  local_handler: Any = _local_handler,
  dispatcher_cls: type[ToolDispatcher] = ToolDispatcher,
) -> ToolDispatcher:
  return dispatcher_cls(
    mcp_client=_NoMcp(),
    local_tool_handlers={declaration.identity.logical_name: local_handler},
    needs_approval=needs_approval,
    request_approval=request_approval,
    approved_tool_types=approved,
    role="owner",
    get_tool_definitions=lambda: [{
      "name": declaration.identity.logical_name,
      "input_schema": {"type": "object"},
    }],
    tool_registration_catalog=ToolRegistrationCatalog((declaration,), ()),
    tool_policy_implementations=registry,
    redaction_context_factory=redaction_context_factory,
    approval_predicate_context_factory=context_factory,
    registered_approval_overlay=overlay,
  )


def test_registered_local_redaction_uses_exact_prepared_call_and_declaration() -> None:
  context = object()
  redaction_calls: list[tuple[Any, Any]] = []
  resolved_declarations: list[Any] = []

  def redact(reference: Any, call: Any) -> RedactionResult:
    redaction_calls.append((reference, call))
    return RedactionResult({
      "normalized": call.prepared_input["normalized"],
      "credential": "<registered-redaction>",
    })

  declaration = _declaration(ToolApprovalPolicy("never"))
  dispatcher = _local_dispatcher(
    declaration,
    _registry(redaction=redact),
    needs_approval=lambda *_args: False,
    redaction_context_factory=lambda resolved: (
      resolved_declarations.append(resolved) or context
    ),
  )
  prepared = PreparedToolCall({
    "normalized": True,
    "credential": "raw-secret",
  })

  assert dispatcher.redact_prepared_tool_input(
    "registered_write",
    prepared,
  ) == {
    "normalized": True,
    "credential": "<registered-redaction>",
  }
  assert resolved_declarations == [declaration]
  assert redaction_calls[0][0] == declaration.semantics.redaction_policy
  assert redaction_calls[0][1].identity == declaration.identity
  assert redaction_calls[0][1].trusted_context is context


def test_registered_redaction_uses_generic_owner_without_live_route() -> None:
  declaration = _declaration(ToolApprovalPolicy("never"))
  dispatcher = _local_dispatcher(
    declaration,
    _registry(),
    needs_approval=lambda *_args: False,
  )
  prepared = PreparedToolCall({
    "symbol": "AAPL",
    "credential_note": "not-a-live-route",
  })

  assert dispatcher.redact_prepared_tool_input(
    "unadvertised_tool",
    prepared,
  ) == redact_tool_input_for_event(
    "unadvertised_tool",
    prepared.materialize_input(),
  )


def test_registered_local_history_redaction_uses_raw_input_without_preparation() -> None:
  context = object()
  observed: list[Any] = []

  def redact(reference: Any, call: Any) -> RedactionResult:
    observed.append((reference, call))
    return RedactionResult({
      "credential": "<registered-redaction>",
      "request": call.prepared_input["request"],
    })

  declaration = _declaration(ToolApprovalPolicy("never"))
  dispatcher = _local_dispatcher(
    declaration,
    _registry(redaction=redact),
    needs_approval=lambda *_args: False,
    redaction_context_factory=lambda _declaration: context,
  )
  raw_input = {
    "credential": "raw-secret",
    "request": {
      "value": 7,
      "filters": [{"field": "ticker", "values": ["AAPL", "MSFT"]}],
    },
  }

  projected = dispatcher.redact_raw_tool_input_for_history(
    "registered_write",
    raw_input,
  )
  assert projected == {
    "credential": "<registered-redaction>",
    "request": {
      "value": 7,
      "filters": [{"field": "ticker", "values": ["AAPL", "MSFT"]}],
    },
  }
  assert json.loads(json.dumps(projected)) == projected
  assert type(projected["request"]) is dict
  assert type(projected["request"]["filters"]) is list
  assert type(projected["request"]["filters"][0]) is dict
  assert raw_input["credential"] == "raw-secret"
  reference, call = observed[0]
  assert reference == declaration.semantics.redaction_policy
  assert call.identity == declaration.identity
  assert call.prepared_input["credential"] == "raw-secret"
  assert call.prepared_input["request"]["value"] == 7
  assert call.prepared_input["request"]["filters"][0]["values"] == (
    "AAPL",
    "MSFT",
  )
  assert "trusted_dispatch_scope" not in call.prepared_input
  assert call.trusted_context is context


def test_registered_dispatcher_requires_redaction_owner_at_construction() -> None:
  declaration = _declaration(ToolApprovalPolicy("never"))

  with pytest.raises(
    ValueError,
    match="registered tool catalog requires a redaction context factory",
  ):
    ToolDispatcher(
      mcp_client=_NoMcp(),
      local_tool_handlers={"registered_write": _local_handler},
      tool_registration_catalog=ToolRegistrationCatalog((declaration,), ()),
      tool_policy_implementations=_registry(),
    )


def test_registered_durable_approval_receives_exact_policy_redaction(
  tmp_path: Path,
  owner_session_host_policy,
) -> None:
  captured: dict[str, object] = {}

  class _CapturingDispatcher(ToolDispatcher):
    async def _run_approval_lifecycle(
      self,
      **kwargs: object,
    ) -> dict[str, object]:
      captured.update(kwargs)
      return {
        "approved": True,
        "tool_input": kwargs["tool_input"],
        "request": SimpleNamespace(state="approved"),
      }

  def redact(_reference: Any, call: Any) -> RedactionResult:
    return RedactionResult({
      "normalized": call.prepared_input["normalized"],
      "credential": "<registered-redaction>",
    })

  declaration = _declaration(
    ToolApprovalPolicy("always"),
    name="memory_write",
  )
  dispatcher = _local_dispatcher(
    declaration,
    _registry(redaction=redact),
    needs_approval=lambda *_args: False,
    dispatcher_cls=_CapturingDispatcher,
  )
  session = SessionStore(ttl=3600).create_session(
    api_key_hash="hash",
    user_id="alice",
    role="owner",
  )
  dispatcher._approval_route = DurableLocalApprovalRoute(
    SQLiteApprovalStore(tmp_path / "local-redaction.sqlite3"),
    SingleUserApprovalPolicy(),
    session,
  )
  dispatcher._session = session

  prepared = PreparedToolCall({
    "normalized": True,
    "credential": "raw-secret",
  })
  result = asyncio.run(dispatcher.dispatch_prepared(
    "call-redaction",
    "memory_write",
    prepared,
  ))

  assert result[1] is None, result
  assert result == ({
    "received": {
      "normalized": True,
      "credential": "raw-secret",
    }
  }, None)
  assert captured["approval_args_redacted"] == {
    "normalized": True,
    "credential": "<registered-redaction>",
  }
  assert captured["approval_args_hash"] == hash_approval_arguments(
    prepared.materialize_input()
  )


def test_registered_native_mcp_redaction_delegates_to_live_manager_owner() -> None:
  declaration = _declaration(
    ToolApprovalPolicy("never"),
    route_kind="mcp",
    name="logical_write",
    server_id="logical-server",
  )
  server = RegisteredToolServerDescriptor(
    "logical-server",
    "transport-server",
    30,
    {},
    _ref("session_injection"),
  )
  observed: list[tuple[str, PreparedToolCall]] = []

  class _RegisteredMcp(_NoMcp):
    def is_mcp_tool(self, name: str) -> bool:
      return name == "provider__logical_write"

    def redact_registered_tool_input(
      self,
      exposed_name: str,
      prepared_call: PreparedToolCall,
    ) -> dict[str, object]:
      observed.append((exposed_name, prepared_call))
      return {"credential": "<manager-redaction>"}

  dispatcher = ToolDispatcher(
    mcp_client=_RegisteredMcp(),
    tool_registration_catalog=ToolRegistrationCatalog(
      (declaration,),
      (server,),
    ),
    tool_policy_implementations=_registry(),
    redaction_context_factory=lambda _declaration: (_ for _ in ()).throw(
      AssertionError("native MCP redaction cannot use dispatcher context")
    ),
  )
  prepared = PreparedToolCall({"credential": "raw-secret"})

  assert dispatcher.redact_prepared_tool_input(
    "provider__logical_write",
    prepared,
  ) == {"credential": "<manager-redaction>"}
  assert observed == [("provider__logical_write", prepared)]


def test_registered_native_mcp_history_redaction_delegates_raw_input() -> None:
  declaration = _declaration(
    ToolApprovalPolicy("never"),
    route_kind="mcp",
    name="logical_write",
    server_id="logical-server",
  )
  server = RegisteredToolServerDescriptor(
    "logical-server",
    "transport-server",
    30,
    {},
    _ref("session_injection"),
  )
  observed: list[tuple[str, Mapping[str, object]]] = []

  class _RegisteredMcp(_NoMcp):
    def redact_registered_raw_tool_input(
      self,
      exposed_name: str,
      tool_input: Mapping[str, object],
    ) -> dict[str, object]:
      observed.append((exposed_name, tool_input))
      return {"credential": "<manager-redaction>"}

  dispatcher = ToolDispatcher(
    mcp_client=_RegisteredMcp(),
    tool_registration_catalog=ToolRegistrationCatalog(
      (declaration,),
      (server,),
    ),
    tool_policy_implementations=_registry(),
    redaction_context_factory=lambda _declaration: (_ for _ in ()).throw(
      AssertionError("native MCP history cannot use dispatcher context")
    ),
  )
  raw_input = {"credential": "raw-secret"}

  assert dispatcher.redact_raw_tool_input_for_history(
    "provider__logical_write",
    raw_input,
  ) == {"credential": "<manager-redaction>"}
  assert observed == [("provider__logical_write", raw_input)]


def test_registered_native_mcp_durable_approval_uses_manager_redaction(
  tmp_path: Path,
  owner_session_host_policy,
) -> None:
  declaration = _declaration(
    ToolApprovalPolicy("always"),
    route_kind="mcp",
    name="logical_write",
    server_id="logical-server",
  )
  server = RegisteredToolServerDescriptor(
    "logical-server",
    "logical-server",
    30,
    {},
    _ref("session_injection"),
  )
  observed: list[tuple[str, PreparedToolCall]] = []
  captured: dict[str, object] = {}

  class _CapturingDispatcher(ToolDispatcher):
    async def _run_approval_lifecycle(
      self,
      **kwargs: object,
    ) -> dict[str, object]:
      captured.update(kwargs)
      return {
        "approved": True,
        "tool_input": kwargs["tool_input"],
        "request": SimpleNamespace(state="approved"),
      }

  class _RegisteredMcp(_NoMcp):
    def is_mcp_tool(self, name: str) -> bool:
      return name == "provider__logical_write"

    def uses_registered_tool_catalog(self) -> bool:
      return True

    def get_registered_mcp_tool_descriptor(
      self,
      exposed_name: str,
    ) -> RegisteredMcpToolDescriptor:
      assert exposed_name == "provider__logical_write"
      return _registered_descriptor(exposed_name, declaration, server)

    def redact_registered_tool_input(
      self,
      exposed_name: str,
      prepared_call: PreparedToolCall,
    ) -> dict[str, object]:
      observed.append((exposed_name, prepared_call))
      return {"credential": "<manager-redaction>"}

    def get_server_for_tool(self, name: str) -> str:
      _ = name
      return "logical-server"

    def classify_registered_mcp_prepared_tool_call(
      self,
      exposed_name: str,
      prepared_call: PreparedToolCall,
      trusted_dispatch_scope: object | None,
      registered_approval_overlay: (
        Callable[[ToolRegistrationDeclaration, PreparedToolCall], bool] | None
      ) = None,
    ) -> RegisteredMcpDirectToolCall:
      _ = trusted_dispatch_scope, registered_approval_overlay
      assert exposed_name == "provider__logical_write"
      return RegisteredMcpDirectToolCall(
        _registered_descriptor(exposed_name, declaration, server),
        prepared_call,
        PlanDecision("none"),
        True,
        None,
      )

    async def call_tool(
      self,
      name: str,
      tool_input: object,
      meta: object | None = None,
      abort_event: asyncio.Event | None = None,
      gateway_session: object | None = None,
      allow_uncertain_replay: bool = True,
      trusted_dispatch_scope: object | None = None,
    ) -> ToolResult:
      _ = (
        name,
        tool_input,
        meta,
        abort_event,
        gateway_session,
        allow_uncertain_replay,
        trusted_dispatch_scope,
      )
      return {"status": "ok"}, None

  dispatcher = _CapturingDispatcher(
    mcp_client=_RegisteredMcp(),
    role="owner",
    needs_approval=lambda *_args: False,
    tool_registration_catalog=ToolRegistrationCatalog(
      (declaration,),
      (server,),
    ),
    tool_policy_implementations=_registry(),
    redaction_context_factory=lambda _declaration: (_ for _ in ()).throw(
      AssertionError("native MCP approval cannot use dispatcher context")
    ),
    approval_predicate_context_factory=lambda *_args: None,
  )
  session = SessionStore(ttl=3600).create_session(
    api_key_hash="hash",
    user_id="alice",
    role="owner",
  )
  dispatcher._approval_route = DurableLocalApprovalRoute(
    SQLiteApprovalStore(tmp_path / "native-redaction.sqlite3"),
    SingleUserApprovalPolicy(),
    session,
  )
  dispatcher._session = session

  prepared = PreparedToolCall({"credential": "raw-secret"})
  result = asyncio.run(dispatcher.dispatch_prepared(
    "call-native-mcp-redaction",
    "provider__logical_write",
    prepared,
    advertised_tool_names={"provider__logical_write"},
  ))

  assert result == ({"status": "ok"}, None)
  assert observed == [("provider__logical_write", prepared)]
  assert captured["approval_args_redacted"] == {
    "credential": "<manager-redaction>",
  }
  assert captured["approval_args_hash"] == hash_approval_arguments(
    prepared.materialize_input()
  )


def test_registered_predicate_uses_prepared_call_and_never_legacy_policy() -> None:
  context = object()
  predicate_calls: list[Any] = []
  cache_calls: list[Any] = []

  def predicate(_reference: Any, call: Any) -> bool:
    predicate_calls.append(call)
    return call.prepared_input["write"] is True

  def cache_key(_reference: Any, call: Any) -> str:
    cache_calls.append(call)
    return "safe:prepared-write"

  declaration = _declaration(ToolApprovalPolicy(
    "predicate",
    predicate=_ref("approval_predicate"),
    cache_key=_ref("approval_cache_key"),
  ))
  dispatcher = _local_dispatcher(
    declaration,
    _registry(predicate=predicate, cache_key=cache_key),
    needs_approval=lambda *_args: (_ for _ in ()).throw(
      AssertionError("registered calls cannot consult legacy approval policy")
    ),
    request_approval=lambda _request: None,
    context_factory=lambda _declaration, prepared: (
      context
      if prepared.exact_backend == "sandbox"
      else (_ for _ in ()).throw(AssertionError("exact backend was lost"))
    ),
  )

  assert dispatcher.requires_approval_prepared(
    "registered_write",
    PreparedToolCall({"write": False}, "sandbox"),
  ) is False
  assert dispatcher.requires_approval_prepared(
    "registered_write",
    PreparedToolCall({"write": True}, "sandbox"),
  ) is True
  assert [call.prepared_input["write"] for call in predicate_calls] == [False, True]
  assert predicate_calls[-1].trusted_context is context
  assert cache_calls[-1].exact_backend == "sandbox"


def test_planned_write_timeout_classification_does_not_require_plan_cache_key(
) -> None:
  def cache_key(_reference: Any, _call: Any) -> str:
    raise AssertionError("the exact plan does not exist during classification")

  declaration = _declaration(
    ToolApprovalPolicy(
      "always",
      cache_key=_ref("approval_cache_key", "prepared-plan"),
    ),
    planning_policy=_ref("planning", "change-set"),
  )
  dispatcher = _local_dispatcher(
    declaration,
    _registry(
      cache_key=cache_key,
      cache_key_policy_id="prepared-plan",
      planning_policy_id="change-set",
    ),
    needs_approval=lambda *_args: (_ for _ in ()).throw(
      AssertionError("registered calls cannot consult legacy approval policy")
    ),
    request_approval=lambda _request: None,
  )

  assert dispatcher.requires_approval_prepared(
    "registered_write",
    PreparedToolCall({"value": 1}),
  ) is True


def test_registered_local_outcome_uses_the_exact_declaration_policy() -> None:
  calls: list[tuple[Any, Any]] = []

  def outcome(reference: Any, call: Any) -> str:
    calls.append((reference, call))
    return "error_semantic"

  declaration = _declaration(ToolApprovalPolicy("never"))
  dispatcher = _local_dispatcher(
    declaration,
    _registry(outcome=outcome),
    needs_approval=lambda *_args: False,
  )

  assert dispatcher.settle_tool_result(
    "registered_write",
    None,
    {"status": "success"},
    None,
    prepared_call=PreparedToolCall({"status": "success"}),
  ).outcome == "error_semantic"
  assert calls[0][0] == declaration.semantics.outcome_policy
  assert calls[0][1].result == {"status": "success"}


def test_catalogless_dispatcher_preserves_generic_outcome_behavior() -> None:
  dispatcher = ToolDispatcher(
    mcp_client=_NoMcp(),
    local_tool_handlers={"legacy_read": _local_handler},
  )

  assert dispatcher.settle_tool_result(
    "legacy_read",
    None,
    {"status": "error", "error": {"code": "not_found"}},
    None,
    prepared_call=PreparedToolCall({}),
  ).outcome == "error_semantic"


def test_registered_mcp_outcome_resolves_exact_exposed_route_with_same_names() -> None:
  declaration_a = _declaration(
    ToolApprovalPolicy("never"),
    route_kind="mcp",
    name="logical_read",
    server_id="server-a",
    outcome_policy=_ref("outcome", "outcome-a"),
  )
  declaration_b = _declaration(
    ToolApprovalPolicy("never"),
    route_kind="mcp",
    name="logical_read",
    server_id="server-b",
    outcome_policy=_ref("outcome", "outcome-b"),
  )
  servers = (
    RegisteredToolServerDescriptor(
      "server-a", "server-a", 30, {}, _ref("session_injection")
    ),
    RegisteredToolServerDescriptor(
      "server-b", "server-b", 30, {}, _ref("session_injection")
    ),
  )
  declarations = {
    "provider_a__logical_read": declaration_a,
    "provider_b__logical_read": declaration_b,
  }

  class _RegisteredMcp(_NoMcp):
    def is_mcp_tool(self, name: str) -> bool:
      return name in declarations

    def uses_registered_tool_catalog(self) -> bool:
      return True

    def get_registered_mcp_tool_descriptor(
      self,
      exposed_name: str,
    ) -> RegisteredMcpToolDescriptor:
      declaration = declarations[exposed_name]
      server = servers[0] if declaration is declaration_a else servers[1]
      return _registered_descriptor(exposed_name, declaration, server)

  dispatcher = ToolDispatcher(
    mcp_client=_RegisteredMcp(),
    tool_registration_catalog=ToolRegistrationCatalog(
      (declaration_a, declaration_b),
      servers,
    ),
    tool_policy_implementations=_registry(extra_outcomes=(
      ("outcome-a", lambda _ref, _call: "ok"),
      ("outcome-b", lambda _ref, _call: "error_semantic"),
    )),
    redaction_context_factory=lambda _declaration: None,
  )

  assert dispatcher.settle_tool_result(
    "provider_a__logical_read",
    None,
    {"status": "same"},
    None,
    prepared_call=PreparedToolCall({}),
  ).outcome == "ok"
  assert dispatcher.settle_tool_result(
    "provider_b__logical_read",
    None,
    {"status": "same"},
    None,
    prepared_call=PreparedToolCall({}),
  ).outcome == "error_semantic"


def test_registered_safe_key_is_the_only_installed_and_reused_cache_key() -> None:
  approved = {"registered_write"}
  approval_requests: list[Any] = []
  cache_calls: list[Any] = []

  def cache_key(_reference: Any, call: Any) -> str:
    cache_calls.append(call)
    return "safe:exact-prepared-call"

  async def approve(request: Any) -> ApprovalDecision:
    approval_requests.append(request)
    return ApprovalDecision(approved=True, allow_tool_type=True)

  declaration = _declaration(ToolApprovalPolicy(
    "always",
    cache_key=_ref("approval_cache_key"),
  ))
  dispatcher = _local_dispatcher(
    declaration,
    _registry(cache_key=cache_key),
    needs_approval=lambda *_args: (_ for _ in ()).throw(
      AssertionError("registered calls cannot consult legacy approval policy")
    ),
    request_approval=approve,
    approved=approved,
  )
  prepared = PreparedToolCall({"value": 7}, "sandbox")

  first = asyncio.run(
    dispatcher.dispatch_prepared("call-1", "registered_write", prepared)
  )
  second = asyncio.run(
    dispatcher.dispatch_prepared("call-2", "registered_write", prepared)
  )

  assert first == ({"received": {"value": 7}}, None)
  assert second == first
  assert len(approval_requests) == 1
  assert approved == {"registered_write", "safe:exact-prepared-call"}
  assert all(call.exact_backend == "sandbox" for call in cache_calls)


def test_registered_overlay_adds_approval_and_disables_cache_reuse() -> None:
  approved: set[str] = set()
  approval_requests: list[Any] = []

  async def approve(request: Any) -> ApprovalDecision:
    approval_requests.append(request)
    return ApprovalDecision(approved=True, allow_tool_type=True)

  declaration = _declaration(ToolApprovalPolicy(
    "always",
    cache_key=_ref("approval_cache_key"),
  ))
  dispatcher = _local_dispatcher(
    declaration,
    _registry(),
    needs_approval=lambda *_args: False,
    request_approval=approve,
    approved=approved,
    overlay=lambda _declaration, _prepared: True,
  )
  prepared = PreparedToolCall({"value": 7})

  asyncio.run(dispatcher.dispatch_prepared("call-1", "registered_write", prepared))
  asyncio.run(dispatcher.dispatch_prepared("call-2", "registered_write", prepared))

  assert len(approval_requests) == 2
  assert approved == set()


def test_registered_never_executes_without_legacy_approval_policy() -> None:
  declaration = _declaration(ToolApprovalPolicy(mode="never"))
  dispatcher = _local_dispatcher(
    declaration,
    _registry(),
    needs_approval=lambda *_args: (_ for _ in ()).throw(
      AssertionError("registered calls cannot consult legacy approval policy")
    ),
    request_approval=lambda _request: (_ for _ in ()).throw(
      AssertionError("approval=never cannot request approval")
    ),
  )

  result = asyncio.run(dispatcher.dispatch_prepared(
    "call-1",
    "registered_write",
    PreparedToolCall({"value": 7}),
  ))

  assert result == ({"received": {"value": 7}}, None)


@pytest.mark.parametrize("failure", ("predicate", "cache_key", "overlay"))
def test_registered_policy_failure_stops_before_approval_and_execution(
  failure: str,
) -> None:
  approval_requests: list[Any] = []
  handler_calls: list[dict[str, Any]] = []

  def fail(*_args: Any) -> Any:
    raise RuntimeError("policy unavailable")

  async def approve(request: Any) -> ApprovalDecision:
    approval_requests.append(request)
    return ApprovalDecision(approved=True)

  async def handler(tool_input: dict[str, Any], **_kwargs: Any):
    handler_calls.append(tool_input)
    return {"received": tool_input}, None

  approval = ToolApprovalPolicy(
    "predicate" if failure == "predicate" else "always",
    predicate=(
      _ref("approval_predicate") if failure == "predicate" else None
    ),
    cache_key=_ref("approval_cache_key"),
  )
  dispatcher = _local_dispatcher(
    _declaration(approval),
    _registry(
      predicate=fail if failure == "predicate" else (lambda _ref, _call: True),
      cache_key=fail if failure == "cache_key" else (lambda _ref, _call: "safe"),
    ),
    needs_approval=lambda *_args: (_ for _ in ()).throw(
      AssertionError("registered calls cannot consult legacy approval policy")
    ),
    request_approval=approve,
    overlay=fail if failure == "overlay" else None,
    local_handler=handler,
  )
  prepared = PreparedToolCall({"value": 1})

  assert dispatcher.requires_approval_prepared(
    "registered_write",
    prepared,
  ) is True
  result, error = asyncio.run(
    dispatcher.dispatch_prepared("call-1", "registered_write", prepared)
  )

  assert result is None
  assert error == {
    "code": "registered_approval_policy_failed",
    "message": "Tool 'registered_write' approval policy could not be evaluated.",
  }
  assert approval_requests == []
  assert handler_calls == []


def test_catalog_without_registered_approval_runtime_preserves_legacy_policy() -> None:
  declaration = _declaration(ToolApprovalPolicy(mode="never"))
  legacy_calls: list[tuple[str, dict[str, Any], str]] = []

  def legacy(tool_name: str, tool_input: dict[str, Any], qualifier: str) -> bool:
    legacy_calls.append((tool_name, tool_input, qualifier))
    return True

  dispatcher = _local_dispatcher(
    declaration,
    _registry(),
    needs_approval=legacy,
    request_approval=lambda _request: None,
    context_factory=None,
  )

  assert dispatcher.requires_approval_prepared(
    "registered_write",
    PreparedToolCall({"value": 1}),
  ) is True
  assert legacy_calls == [("registered_write", {"value": 1}, "")]


def test_live_registered_mcp_resolves_its_descriptor_not_its_exposed_name() -> None:
  declaration = _declaration(
    ToolApprovalPolicy(mode="always"),
    route_kind="mcp",
    name="logical_write",
    server_id="logical-server",
  )
  server = RegisteredToolServerDescriptor(
    "logical-server",
    "logical-server",
    30,
    {},
    _ref("session_injection"),
  )
  catalog = ToolRegistrationCatalog((declaration,), (server,))

  class _RegisteredMcp(_NoMcp):
    def is_mcp_tool(self, name: str) -> bool:
      return name == "provider__logical_write"

    def uses_registered_tool_catalog(self) -> bool:
      return True

    def get_registered_mcp_tool_descriptor(
      self,
      exposed_name: str,
    ) -> RegisteredMcpToolDescriptor:
      assert exposed_name == "provider__logical_write"
      return _registered_descriptor(exposed_name, declaration, server)

  async def _request_approval(
    _request: TransportApprovalRequest,
  ) -> None:
    return None

  dispatcher = ToolDispatcher(
    mcp_client=_RegisteredMcp(),
    needs_approval=lambda *_args: (_ for _ in ()).throw(
      AssertionError("registered calls cannot consult legacy approval policy")
    ),
    request_approval=_request_approval,
    tool_registration_catalog=catalog,
    tool_policy_implementations=_registry(),
    redaction_context_factory=lambda _declaration: None,
    approval_predicate_context_factory=lambda _declaration, _prepared: None,
  )

  assert dispatcher.requires_approval_prepared(
    "provider__logical_write",
    PreparedToolCall({"value": 1}),
  ) is True
