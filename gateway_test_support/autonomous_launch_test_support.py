from __future__ import annotations

from typing import Literal, TypedDict
from agent_gateway.autonomous_launch_envelope import (
  AUTONOMOUS_CAPABILITY_ENVELOPE_AUDIENCE,
  AUTONOMOUS_CAPABILITY_ENVELOPE_VERSION,
  AUTONOMOUS_RUNTIME_SESSION_PURPOSE,
  AutonomousControlAuthority,
  AutonomousDispatchScope,
  AutonomousLaunchEnvelope,
  AutonomousLaunchWorkload,
  AutonomousSessionAuthority,
  OrdinaryAutonomousSessionAuthority,
  sign_autonomous_launch_envelope,
  verify_autonomous_launch_envelope,
)
from agent_gateway.capability_binding import (
  CapabilityBind,
  CredentialHandle,
  CredentialPrincipal,
  RunMode,
)
from agent_gateway.skill_limits import SkillExecutionLimits
from agent_gateway.agent_session_log_layout import (
  AutonomousSessionLogAuthority,
)

def _control_authority(
  *,
  admission_ledger_path: str = "/tmp/autonomous-admissions.sqlite3",
  operator_inbox_path: str = "/tmp/bg_7.operator-messages.jsonl",
) -> AutonomousControlAuthority:
  return AutonomousControlAuthority(
    control_mode="file",
    admission_ledger_path=admission_ledger_path,
    admission_ledger_device=1,
    admission_ledger_inode=10,
    operator_inbox_path=operator_inbox_path,
    operator_inbox_device=1,
    operator_inbox_inode=11,
  )

def _memory_control_authority() -> AutonomousControlAuthority:
  return AutonomousControlAuthority(
    control_mode="memory",
    admission_ledger_path=None,
    admission_ledger_device=None,
    admission_ledger_inode=None,
    operator_inbox_path=None,
    operator_inbox_device=None,
    operator_inbox_inode=None,
  )

def _workload(
  *,
  profile: str = "analyst",
  mode: Literal["run_once", "task", "skill", "pack"] = "run_once",
  task: str | None = None,
  skill: str | None = None,
  pack: str | None = None,
  context: str | None = None,
  ticker: str | None = None,
  dev_mode: bool = False,
  max_budget_usd: float | None = None,
  deliver: bool = True,
  admitted_skill_execution_limits: SkillExecutionLimits | None = None,
) -> AutonomousLaunchWorkload:
  if mode == "skill" and admitted_skill_execution_limits is None:
    admitted_skill_execution_limits = SkillExecutionLimits(
      20,
      32_000,
      20.0,
    )
  return AutonomousLaunchWorkload(
    profile=profile,
    mode=mode,
    task=task,
    skill=skill,
    pack=pack,
    context=context,
    ticker=ticker,
    research_file_id=None,
    dev_mode=dev_mode,
    max_budget_usd=max_budget_usd,
    deliver=deliver,
    admitted_skill_execution_limits=admitted_skill_execution_limits,
    session_log_authority=AutonomousSessionLogAuthority(
      layout="v1",
      provider_session_epoch=None,
      base_path="/tmp/autonomous-session-logs",
      root_path=None,
      root_device=None,
      root_inode=None,
      active_path=None,
      active_device=None,
      active_inode=None,
      meta_path=None,
      meta_device=None,
      meta_inode=None,
      storage_identity_digest=None,
    ),
  )

def _bind(
  *,
  principal: CredentialPrincipal = "service",
  run_mode: RunMode = "autonomous",
) -> CapabilityBind:
  return CapabilityBind(
    schema_version="1.0",
    capability_id="session.driver",
    model_key="anthropic.test-autonomous",
    provider="anthropic",
    upstream_model="claude-test",
    adapter="anthropic.messages",
    protocol_profile="messages.standard",
    route="anthropic.public",
    effort="high",
    credential_principal=principal,
    credential_ref=(
      "autonomous-user:test-handle"
      if principal == "user"
      else "autonomous-service:test-handle"
    ),
    run_mode=run_mode,
    registry_revision="test-registry-1",
    policy_revision="test-policy-1",
    selection_source="capability_default",
  )

def _credential_handle(
  *,
  tenant_id: str = "tenant-ordinary",
  principal: CredentialPrincipal = "user",
) -> CredentialHandle:
  return CredentialHandle(
    handle_id=(
      "autonomous-user:test-handle"
      if principal == "user"
      else "autonomous-service:test-handle"
    ),
    provider="anthropic",
    principal=principal,
    tenant_id=tenant_id,
    actor_id="42" if principal == "user" else None,
  )

def _ordinary_session_authority(
  *,
  bind: CapabilityBind | None = None,
  dispatch_scope: AutonomousDispatchScope | None = None,
  role: str = "owner",
) -> AutonomousSessionAuthority:
  resolved_bind = bind or _bind()
  handle = _credential_handle(
    tenant_id="tenant-ordinary",
    principal=resolved_bind.credential_principal,
  )
  return AutonomousSessionAuthority.ordinary(
    OrdinaryAutonomousSessionAuthority(
      session_id="bg_7",
      tenant_id="tenant-ordinary",
      user_id="42",
      owner_user_id="42",
      created_at=1_799_999_900,
      expires_at=1_800_000_600,
      user_email="owner@example.test",
      risk_user_id=7,
      role=role,
      kind="chat",
      channel="cli",
      purpose=AUTONOMOUS_RUNTIME_SESSION_PURPOSE,
      raw_user_id="42",
      user_slug="owner",
      user_aliases=("42", "owner", "owner@example.test"),
      identity_status="canonical",
      schema_version=1,
      is_public=False,
      allow_service_for_interactive=False,
      auth_provider=resolved_bind.provider,
      credential_handle=handle,
    ),
    dispatch_scope=dispatch_scope,
  )
