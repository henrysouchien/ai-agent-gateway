from __future__ import annotations

import asyncio
import builtins
from importlib import metadata
import os
import subprocess
import sys
import textwrap
import tomllib
from pathlib import Path
from typing import Any
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

from agent_gateway.approval_audit import ApprovalAuditEmitter, build_audit_entry
from agent_gateway.approval_policy import ApprovalRequest, utc_now
from agent_gateway.model_registry import (
  INITIAL_MODEL_REGISTRY,
  INITIAL_MODEL_SELECTION_POLICY,
)
from agent_gateway.event_log import EventLog
from agent_gateway.server import ChatRuntime, GatewayServerConfig, create_gateway_app
from agent_gateway.server_models import BuildChatRuntime, SystemPrompt
from agent_gateway.secret_boundary import SecretBoundary


class _NoopRunner:
  async def run(
    self,
    *,
    messages: list[dict[str, object]],
    system_prompt: SystemPrompt | None = None,
    max_turns: int | None = None,
  ) -> None:
    _ = messages, system_prompt, max_turns


def _build_noop_runner(
  _event_log: EventLog,
  _session_id: str,
  _started_at: float,
) -> _NoopRunner:
  return _NoopRunner()


async def _build_chat_runtime_impl(
  session,
  request,
  channel,
  auth_manager,
  *,
  storage_root: Path | None = None,
):
  _ = session, channel, auth_manager, storage_root
  return ChatRuntime(
    system_prompt="test",
    build_runner=_build_noop_runner,
    capability_execution=request.capability_execution,
  )


_build_chat_runtime: BuildChatRuntime = _build_chat_runtime_impl


def test_all_modules_import_without_checkout_trees(tmp_path: Path) -> None:
  package_dir = Path(__file__).resolve().parents[1]
  dependency_dir = tmp_path / "dependencies"
  dependency_dir.mkdir()
  manifest = tomllib.loads((package_dir / "pyproject.toml").read_text())
  pending = [Requirement(value) for value in manifest["project"]["dependencies"]]
  visited: set[tuple[str, frozenset[str]]] = set()
  while pending:
    requirement = pending.pop()
    key = (canonicalize_name(requirement.name), frozenset(requirement.extras))
    if key in visited:
      continue
    visited.add(key)
    distribution = metadata.distribution(requirement.name)
    # Expose only the distributions in the declared dependency closure, not
    # site-packages wholesale (which can conceal another checkout reach).
    for file in distribution.files or ():
      top = file.parts[0]
      if top in {".", ".."} or top.endswith(".pth"):
        continue
      target = dependency_dir / top
      if not target.exists():
        # `locate_file` is typed as the `SimplePath` protocol; every installed
        # distribution here resolves it to a real filesystem path.
        source = Path(str(distribution.locate_file(top)))
        if source.exists():
          target.symlink_to(source, target_is_directory=source.is_dir())
    for value in distribution.requires or ():
      dependency = Requirement(value)
      if dependency.marker is None or any(
        dependency.marker.evaluate({"extra": extra})
        for extra in requirement.extras or {""}
      ):
        pending.append(dependency)
  script = textwrap.dedent(
    """
    import importlib
    import importlib.abc
    import pkgutil
    import sys
    forbidden = {
      "api", "agent", "schema", "memory", "research", "mcp_servers",
      "investment_tools", "scripts", "fms", "user_identity",
    }

    class CheckoutImportGuard(importlib.abc.MetaPathFinder):
      def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".", 1)[0] in forbidden:
          raise AssertionError(f"gateway imported checkout module {fullname!r}")

    sys.meta_path.insert(0, CheckoutImportGuard())
    import agent_gateway
    names = sorted(
      module.name
      for module in pkgutil.walk_packages(
        agent_gateway.__path__, prefix="agent_gateway."
      )
    )
    for name in names:
      importlib.import_module(name)
    assert names
    assert agent_gateway.__version__
    print(f"Imported {len(names)} gateway modules without checkout trees")
    """
  )
  env = os.environ.copy()
  env["PYTHONPATH"] = os.pathsep.join((str(package_dir), str(dependency_dir)))
  env.pop("PRODUCT_ID", None)

  result = subprocess.run(
    [sys.executable, "-S", "-c", script],
    cwd=tmp_path,
    env=env,
    capture_output=True,
    text=True,
    check=False,
  )

  assert result.returncode == 0, result.stdout + result.stderr


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
