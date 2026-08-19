from __future__ import annotations

import asyncio
import builtins
import os
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Any

from agent_gateway.approval_audit import ApprovalAuditEmitter, build_audit_entry
from agent_gateway.approval_policy import ApprovalRequest, utc_now
from agent_gateway.model_registry import (
  INITIAL_MODEL_REGISTRY,
  INITIAL_MODEL_SELECTION_POLICY,
)
from agent_gateway.server import ChatRuntime, GatewayServerConfig, create_gateway_app
from agent_gateway.secret_boundary import SecretBoundary


async def _build_chat_runtime(*, session, request, channel, auth_manager):
  _ = session, channel, auth_manager
  return ChatRuntime(
    system_prompt="test",
    build_runner=lambda *_args: None,
    capability_execution=request.capability_execution,
  )


def test_leaf_imports_do_not_require_monorepo_schema(tmp_path: Path) -> None:
  package_dir = Path(__file__).resolve().parents[1]
  script = textwrap.dedent(
    """
    import importlib
    import importlib.util

    if importlib.util.find_spec("schema") is not None:
      raise SystemExit("schema unexpectedly importable before agent_gateway import")

    import agent_gateway
    from agent_gateway.code_execution import CodeExecutionConfig
    from agent_gateway.session import GatewaySession

    if importlib.util.find_spec("schema") is not None:
      raise SystemExit("schema unexpectedly importable after leaf imports")

    assert CodeExecutionConfig is not None
    assert GatewaySession is not None
    assert agent_gateway.__version__
    """
  )
  env = os.environ.copy()
  env["PYTHONPATH"] = str(package_dir)
  env.pop("PRODUCT_ID", None)

  result = subprocess.run(
    [sys.executable, "-c", script],
    cwd=tmp_path,
    env=env,
    capture_output=True,
    text=True,
    check=False,
  )

  assert result.returncode == 0, result.stderr


def test_create_agent_does_not_require_monorepo_schema(tmp_path: Path) -> None:
  package_dir = Path(__file__).resolve().parents[1]
  script = textwrap.dedent(
    """
    import importlib.util

    if importlib.util.find_spec("schema") is not None:
      raise SystemExit("schema unexpectedly importable before create_agent import")

    from agent_gateway import create_agent

    app = create_agent("test")
    if importlib.util.find_spec("schema") is not None:
      raise SystemExit("schema unexpectedly importable after create_agent app build")

    assert app.routes
    """
  )
  env = os.environ.copy()
  env["PYTHONPATH"] = str(package_dir)
  env.pop("PRODUCT_ID", None)

  result = subprocess.run(
    [sys.executable, "-c", script],
    cwd=tmp_path,
    env=env,
    capture_output=True,
    text=True,
    check=False,
  )

  assert result.returncode == 0, result.stderr


def test_create_gateway_app_does_not_import_monorepo_agent_modules(monkeypatch) -> None:
  real_import = builtins.__import__

  def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
    if name == "agent" or name.startswith("agent.") or name == "api" or name.startswith("api."):
      raise AssertionError(f"agent_gateway package imported monorepo module {name!r}")
    return real_import(name, globals, locals, fromlist, level)

  monkeypatch.setattr(builtins, "__import__", guarded_import)

  app = create_gateway_app(
    GatewayServerConfig(
      jwt_secret="package-boundary-test-secret-012345",
      valid_api_keys={"test-key"},
      tenant_id="test-product",
      model_registry=INITIAL_MODEL_REGISTRY,
      model_selection_policy=INITIAL_MODEL_SELECTION_POLICY,
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


def test_approval_audit_emitter_sanitizes_written_entry_after_raw_redactor_input() -> None:
  secret = "CUSTOM-ACTIVE-CREDENTIAL-APPROVAL-AUDIT-8f21d7"
  raw_seen: list[dict[str, Any]] = []
  written: list[Any] = []

  def passthrough_redactor(
    _tool_name: str,
    tool_input: dict[str, Any],
    **_kwargs: Any,
  ) -> dict[str, Any]:
    raw_seen.append(dict(tool_input))
    return dict(tool_input)

  class Writer:
    async def write(self, entry: Any) -> None:
      written.append(entry)

  request = ApprovalRequest(
    approval_id="approval-secret",
    request_id="request-secret",
    tool_call_id="tool-secret",
    parent_approval_id=None,
    approval_chain_id="approval-secret",
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
    requested_at=utc_now(),
    policy_id="test-policy",
    policy_version="1",
    policy_bundle_hash="bundle",
  )
  boundary = SecretBoundary((secret,))
  raw_args = {
    "credential": secret,
    "api_key_set": True,
    "path": "/Users/alice/Documents/report.xlsx",
  }
  emitter = ApprovalAuditEmitter(
    writer=Writer(),
    deployment_secret=b"test-secret",
    key_id="test-key",
    tool_input_redactor=passthrough_redactor,
  )

  asyncio.run(
    emitter.emit_execution_outcome(
      request=request,
      raw_tool_args=raw_args,
      outcome="tool_error",
      error_summary=f"failed {secret}",
      boundary_sanitizer=lambda value, sink: boundary.sanitize(
        value,
        sink=sink,
      ),
    )
  )

  assert raw_seen == [raw_args]
  assert len(written) == 1
  assert written[0].tool_args_redacted == {
    **raw_args,
    "credential": "<redacted-secret>",
  }
  assert written[0].error_summary == "failed <redacted-secret>"
