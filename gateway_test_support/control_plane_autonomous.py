from __future__ import annotations

import asyncio
from itertools import count
import json
import os
from pathlib import Path
from typing import Any

from agent_gateway import AgentRunner
from agent_gateway.autonomous_capability_handoff import AutonomousCapabilityBinding
from agent_gateway.autonomous_event_channel import adopt_inherited_autonomous_event_channel
from agent_gateway.autonomous_launch_envelope import AUTONOMOUS_CAPABILITY_ENVELOPE_ENV
from model_authority.bind import CapabilityBind
from model_authority.binding import CredentialHandle
from agent_gateway.capability_execution import MaterializedCredential
from agent_gateway.claim_signing_authority import GatewayClaimSigningAuthority
from agent_gateway.event_log import EventLog
from model_authority.current import INITIAL_MODEL_REGISTRY, INITIAL_MODEL_SELECTION_POLICY
from agent_gateway.server import ChatRuntime, GatewayServerConfig, create_gateway_app
from agent_gateway.skill_limits import AutonomousSkillAdmissionPolicy, SkillExecutionLimits

from .control_plane_identity import fake_identity_resolver, fake_mcp_user_key_lookup

API_KEY = "autonomous-pr5a-key"
HMAC_KEY = "autonomous-pr5a-hmac-key-at-least-32-bytes"
API_DIR = Path(__file__).resolve().parent
_UNSET = object()
_MODEL_ENTRY = INITIAL_MODEL_REGISTRY.require("anthropic.claude-opus-5")
_SERVICE_HANDLE = CredentialHandle(
  handle_id="service:test-product:anthropic",
  provider="anthropic",
  principal="service",
  tenant_id="test-product",
  actor_id=None,
)
def _autonomous_capability_binding(request) -> AutonomousCapabilityBinding:
  return AutonomousCapabilityBinding(
    bind=request.required_bind
    or CapabilityBind(
      schema_version="1.0",
      capability_id="session.driver",
      model_key=_MODEL_ENTRY.key,
      provider=_MODEL_ENTRY.provider,
      upstream_model=_MODEL_ENTRY.upstream_model,
      adapter=_MODEL_ENTRY.adapter,
      protocol_profile=_MODEL_ENTRY.protocol_profile,
      route=_MODEL_ENTRY.route,
      effort="high",
      credential_principal="service",
      credential_ref=_SERVICE_HANDLE.handle_id,
      run_mode=request.run_mode,
      registry_revision=INITIAL_MODEL_REGISTRY.revision,
      policy_revision=INITIAL_MODEL_SELECTION_POLICY.revision,
      selection_source="capability_default",
    ),
    materialized_credential=MaterializedCredential(
      handle=_SERVICE_HANDLE,
      auth_config={
        "provider": _SERVICE_HANDLE.provider,
        "auth_mode": "api",
        "api_key": "autonomous-pr5a-test-secret",
      },
    ),
  )
_FAKE_PROCESS_PIDS = count(90_000)
_FAKE_PROCESSES: dict[int, "_FakeAutonomousProcess"] = {}


class _FakeStdin:
  def __init__(self) -> None:
    self.buffer = bytearray()

  def write(self, payload: bytes) -> None:
    self.buffer.extend(payload)

  async def drain(self) -> None:
    return None

  def close(self) -> None:
    return None

  async def wait_closed(self) -> None:
    return None


class _FakeAutonomousProcess:
  def __init__(self, inherited_event_fd: int, *, channel_id: str) -> None:
    self.pid = next(_FAKE_PROCESS_PIDS)
    self._returncode: int | None = None
    self._inherited_fds: list[int] = []
    self._event_channel = adopt_inherited_autonomous_event_channel(
      inherited_event_fd,
      channel_id=channel_id,
    )
    self._event_channel.start(timeout_seconds=2)
    self.stdin = _FakeStdin()
    _FAKE_PROCESSES[self.pid] = self

  @property
  def returncode(self) -> int | None:
    return self._returncode

  @returncode.setter
  def returncode(self, value: int | None) -> None:
    self._returncode = value
    if value is not None:
      self._close_inherited_fds()

  def _close_inherited_fds(self) -> None:
    self._event_channel.interrupt()
    for inherited_fd in self._inherited_fds:
      if inherited_fd >= 0:
        os.close(inherited_fd)
    self._inherited_fds.clear()

  async def wait(self) -> int:
    while self.returncode is None:
      await asyncio.sleep(0.01)
    return self.returncode

  def retain_inherited_fds(self, inherited_fds: tuple[int, ...]) -> None:
    self._inherited_fds.extend(os.dup(fd) for fd in inherited_fds)

  def terminate(self) -> None:
    if self.returncode is None:
      self.returncode = -15

  def kill(self) -> None:
    if self.returncode is None:
      self.returncode = -9
class _NoopRunner(AgentRunner):
  def __init__(self) -> None:
    pass

  def bind_selected_content(self, bindings) -> None:
    _ = bindings

  async def run(
    self,
    messages,
    system_prompt=None,
    max_turns=None,
    *,
    resume_initial_messages=None,
  ) -> None:
    _ = messages, system_prompt, max_turns, resume_initial_messages


def _make_app(
  monkeypatch,
  tmp_path: Path,
  *,
  control_skills_dir: Path | None = None,
  control_skill_catalog: Any | None = None,
  control_profile_names_provider: Any | None = None,
  control_profile_loader: Any | None = None,
  admission_policy_resolver: Any = _UNSET,
  dispatch_scope_validator: Any | None = None,
  claim_signing_authority_installed: bool = True,
  autonomous_api_dir: Path | None = API_DIR,
  identity_resolver=fake_identity_resolver,
  mcp_user_key_lookup=fake_mcp_user_key_lookup,
):
  monkeypatch.setenv("AGENT_API_USER_CLAIM_HMAC_KEY", HMAC_KEY)
  monkeypatch.setenv("AGENT_GATEWAY_AUTONOMOUS_LOG_DIR", str(tmp_path / "autonomous-logs"))

  async def _build_chat_runtime(session, request, channel, auth_manager, *, storage_root: Path | None = None):
    _ = session, channel, auth_manager
    return ChatRuntime(
      system_prompt="system",
      build_runner=lambda event_log, _sid, _started_at: _runner_with_log(event_log),
      capability_execution=request.capability_execution,
    )

  if admission_policy_resolver is _UNSET:
    default_admission_resolver = (
      None
      if control_skills_dir is not None
      else lambda skill_name: AutonomousSkillAdmissionPolicy(
        False,
        SkillExecutionLimits(None, None, None),
      )
    )
  else:
    default_admission_resolver = admission_policy_resolver

  return create_gateway_app(
    GatewayServerConfig(
      jwt_secret="autonomous-pr5a-test-secret-0123456789",
      valid_api_keys={API_KEY},
      tenant_id="test-product",
      model_registry=INITIAL_MODEL_REGISTRY,
      model_selection_policy=INITIAL_MODEL_SELECTION_POLICY,
      build_chat_runtime=_build_chat_runtime,
      autonomous_capability_binding_resolver=_autonomous_capability_binding,
      autonomous_skill_admission_policy_resolver=(
        default_admission_resolver
      ),
      autonomous_api_dir=autonomous_api_dir,
      control_skills_dir=control_skills_dir,
      control_skill_catalog=control_skill_catalog,
      control_profile_names_provider=control_profile_names_provider,
      control_profile_loader=control_profile_loader,
      dispatch_scope_validator=dispatch_scope_validator,
      identity_resolver=identity_resolver,
      mcp_user_key_lookup=mcp_user_key_lookup,
      claim_signing_authority=(
        GatewayClaimSigningAuthority(HMAC_KEY)
        if claim_signing_authority_installed
        else None
      ),
    )
  )
def _runner_with_log(_event_log: EventLog) -> _NoopRunner:
  return _NoopRunner()
def _install_fake_spawn(
  monkeypatch,
  *,
  invocations: list[dict[str, Any]] | None = None,
) -> tuple[list[_FakeAutonomousProcess], list[dict[str, str]]]:
  from agent_gateway import autonomous_runner

  processes: list[_FakeAutonomousProcess] = []
  envs: list[dict[str, str]] = []

  async def fake_exec(*args, **kwargs):
    _ = args
    envelope = json.loads(
      kwargs["env"][AUTONOMOUS_CAPABILITY_ENVELOPE_ENV]
    )
    process = _FakeAutonomousProcess(
      os.dup(kwargs["pass_fds"][0]),
      channel_id=envelope["channel_id"],
    )
    process.retain_inherited_fds(tuple(kwargs["pass_fds"][1:]))
    processes.append(process)
    envs.append(dict(kwargs["env"]))
    if invocations is not None:
      invocations.append(dict(kwargs))
    return process

  monkeypatch.setattr(autonomous_runner.asyncio, "create_subprocess_exec", fake_exec)
  monkeypatch.setattr(autonomous_runner, "_get_process_group_id", lambda pid: pid)

  def signal_process_group(process_group_id: int, signal_number: int) -> None:
    _FAKE_PROCESSES[process_group_id].returncode = -signal_number

  monkeypatch.setattr(autonomous_runner, "_signal_process_group", signal_process_group)
  return processes, envs
