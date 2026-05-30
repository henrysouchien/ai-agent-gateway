from __future__ import annotations

import builtins
from typing import Any

from agent_gateway.approval_audit import build_audit_entry
from agent_gateway.approval_policy import ApprovalRequest, utc_now
from agent_gateway.server import ChatRuntime, GatewayServerConfig, create_gateway_app


async def _build_chat_runtime(*, session, request, channel, auth_manager):
  _ = session, request, channel, auth_manager
  return ChatRuntime(system_prompt="test", build_runner=lambda *_args: None)


def test_create_gateway_app_does_not_import_monorepo_agent_modules(monkeypatch) -> None:
  real_import = builtins.__import__

  def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
    if name == "agent" or name.startswith("agent.") or name == "api" or name.startswith("api."):
      raise AssertionError(f"agent_gateway package imported monorepo module {name!r}")
    return real_import(name, globals, locals, fromlist, level)

  monkeypatch.setattr(builtins, "__import__", guarded_import)

  app = create_gateway_app(
    GatewayServerConfig(
      jwt_secret="package-boundary-test-secret",
      valid_api_keys={"test-key"},
      allowed_models=set(),
      build_chat_runtime=_build_chat_runtime,
    )
  )

  assert app.state.gateway_approval_audit_emitter is not None
  assert app.state.gateway_approval_store is not None


def test_build_audit_entry_uses_injected_tool_redactor() -> None:
  calls: list[tuple[str, dict[str, Any]]] = []

  def redactor(tool_name: str, tool_input: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
    calls.append((tool_name, dict(tool_input)))
    assert kwargs["deployment_secret"] == b"test-secret"
    assert kwargs["key_id"] == "test-key"
    return {"safe": True}

  request = ApprovalRequest(
    approval_id="approval-1",
    request_id="request-1",
    tool_call_id="tool-1",
    parent_approval_id=None,
    approval_chain_id="approval-1",
    user_id="user-1",
    profile="analyst",
    channel="cli",
    session_id="session-1",
    run_id=None,
    tool_name="dangerous_tool",
    tool_class="state_write",
    tool_args_redacted={},
    args_hash="",
    reason="test",
    blast_radius_summary="test",
    state="created",
    decider_id=None,
    decider_role=None,
    decision_reason=None,
    requested_at=utc_now(),
    expires_at=None,
    policy_id="test-policy",
    policy_version="1",
    policy_bundle_hash="bundle",
    model_id=None,
    model_version=None,
    system_prompt_hash=None,
    tool_schema_version=None,
    mcp_server_version=None,
    tenant_id=None,
  )

  entry = build_audit_entry(
    raw_tool_args={"secret": "raw"},
    deployment_secret=b"test-secret",
    key_id="test-key",
    event_type="request_created",
    request=request,
    tool_input_redactor=redactor,
  )

  assert calls == [("dangerous_tool", {"secret": "raw"})]
  assert entry.tool_args_redacted == {"safe": True}
