from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from typing import Awaitable, Callable, Literal, TypeAlias

from model_authority.bind import CapabilityBind
from model_authority.binding import RunMode
from model_authority.capabilities import SESSION_DRIVER_CAPABILITY
from agent_workflow_contracts import OperationUnavailable, ResolvedAuthority
from .admission import AdmittedContext, ParentContext, Refusal
from .capability_execution import MaterializedCredential


AutonomousCapabilityBindingSource: TypeAlias = Literal["start", "resume", "schedule"]


def _required_text(value: object, *, field_name: str) -> str:
  text = str(value or "").strip()
  if not text:
    raise ValueError(f"{field_name} must be non-empty")
  return text


@dataclass(frozen=True, slots=True)
class AutonomousCapabilityBindingRequest:
  """Trusted server request for one exact autonomous launch binding."""

  task_id: str
  control_run_id: str
  owner_user_id: str
  raw_user_id: str
  user_email: str | None
  profile: str
  mode: str
  skill: str | None
  channel: str | None
  source: AutonomousCapabilityBindingSource
  run_mode: RunMode
  required_bind: CapabilityBind | None = None
  parent: ParentContext | None = None
  task: str | None = None
  ticker: str | None = None
  research_file_id: int | None = None
  env_names: frozenset[str] = frozenset()
  # The launch edge's user facts beside ``owner_user_id``/``raw_user_id``/
  # ``user_email``; an admitting resolver hands them to ``admit`` as the
  # child's identity.
  role: str | None = None
  user_slug: str | None = None
  risk_user_id: int = 0
  user_aliases: tuple[str, ...] = ()
  identity_status: str | None = None
  # A published preset name a fresh skill launch admits under (design §6.4);
  # a resume's ``required_bind`` never reselects, so it never carries one.
  model_preset: str | None = None

  def __post_init__(self) -> None:
    object.__setattr__(self, "task_id", _required_text(self.task_id, field_name="task_id"))
    object.__setattr__(
      self,
      "control_run_id",
      _required_text(self.control_run_id, field_name="control_run_id"),
    )
    object.__setattr__(
      self,
      "owner_user_id",
      _required_text(self.owner_user_id, field_name="owner_user_id"),
    )
    object.__setattr__(
      self,
      "raw_user_id",
      _required_text(self.raw_user_id, field_name="raw_user_id"),
    )
    object.__setattr__(self, "profile", _required_text(self.profile, field_name="profile"))
    object.__setattr__(self, "mode", _required_text(self.mode, field_name="mode").lower())
    if self.user_email is not None:
      object.__setattr__(
        self,
        "user_email",
        _required_text(self.user_email, field_name="user_email"),
      )
    if self.skill is not None:
      object.__setattr__(self, "skill", _required_text(self.skill, field_name="skill"))
    if self.channel is not None:
      object.__setattr__(
        self,
        "channel",
        _required_text(self.channel, field_name="channel").lower(),
      )
    if self.model_preset is not None:
      object.__setattr__(
        self,
        "model_preset",
        _required_text(self.model_preset, field_name="model_preset"),
      )
    if self.source not in {"start", "resume", "schedule"}:
      raise ValueError(f"unsupported autonomous capability binding source: {self.source!r}")
    if self.run_mode not in {"autonomous", "cron"}:
      raise ValueError("autonomous capability binding run_mode must be 'autonomous' or 'cron'")
    if self.source == "resume" and self.required_bind is None:
      raise ValueError("resume binding requests require an exact persisted bind")
    if self.required_bind is not None:
      if not isinstance(self.required_bind, CapabilityBind):
        raise TypeError("required_bind must be CapabilityBind")


@dataclass(frozen=True, slots=True)
class AutonomousCapabilityBinding:
  """Exact launch binding plus its sole memory-only credential material."""

  bind: CapabilityBind
  materialized_credential: MaterializedCredential = field(
    repr=False,
    compare=False,
  )
  admitted: AdmittedContext[ResolvedAuthority] | None = None

  def __post_init__(self) -> None:
    if not isinstance(self.bind, CapabilityBind):
      raise TypeError("autonomous capability binding bind must be CapabilityBind")
    if self.bind.run_mode not in {"autonomous", "cron"}:
      raise ValueError(
        "autonomous capability binding requires autonomous or cron run mode"
      )
    if self.admitted is not None and self.bind != self.admitted.wire.capability_bind:
      raise ValueError("autonomous capability binding does not match admitted context")
    materialized = self.materialized_credential
    if not isinstance(materialized, MaterializedCredential):
      raise ValueError(
        "autonomous binding requires exact materialized credential"
      )
    handle = materialized.handle
    if (
      handle.handle_id != self.bind.credential_ref
      or handle.principal != self.bind.credential_principal
      or handle.provider != self.bind.provider
    ):
      raise ValueError(
        "autonomous binding credential handle does not match its bind"
      )

class AutonomousLaunchAdmissionRefused(Exception):
  """Surface the admission owner's cause before launch side effects."""

  def __init__(self, refusal: Refusal | OperationUnavailable) -> None:
    self.detail = refusal.detail
    if isinstance(refusal, Refusal):
      self.reason = refusal.reason
      self.owner = refusal.owner
      self.code = refusal.reason
      self.classification = refusal.classification
      self.next_call = refusal.next_call
    else:
      self.reason = None
      self.owner = None
      self.code = "operation_unavailable"
      self.classification = None
      self.next_call = None
    self.http_status = (
      403
      if isinstance(refusal, Refusal) and refusal.reason in {
        "tenant_mismatch", "grant_exceeds_parent", "binding_not_granted",
        "binding_ambiguous", "handle_not_granted", "invite_effect_not_allowed",
      }
      else 422
    )
    super().__init__(self.detail)

  @classmethod
  def from_refusal(cls, refusal: Refusal) -> AutonomousLaunchAdmissionRefused:
    return cls(refusal)

  @classmethod
  def from_unavailable(cls, unavailable: OperationUnavailable) -> AutonomousLaunchAdmissionRefused:
    return cls(unavailable)

  def payload(self) -> dict[str, str]:
    if self.reason is not None and self.owner is not None:
      payload = {"owner": self.owner, "reason": self.reason, "detail": self.detail}
      if self.classification is not None:
        payload["classification"] = self.classification
      if self.next_call is not None:
        payload["next_call"] = self.next_call
      return payload
    return {"code": self.code, "detail": self.detail}



AutonomousCapabilityBindingResolver: TypeAlias = Callable[
  [AutonomousCapabilityBindingRequest],
  AutonomousCapabilityBinding | Awaitable[AutonomousCapabilityBinding],
]


async def resolve_autonomous_capability_binding(
  resolver: AutonomousCapabilityBindingResolver | None,
  request: AutonomousCapabilityBindingRequest,
) -> AutonomousCapabilityBinding:
  """Invoke the trusted resolver once and enforce exact resume semantics."""

  if resolver is None:
    raise RuntimeError(
      "autonomous capability binding resolver is required before subprocess launch"
    )
  resolved = resolver(request)
  if inspect.isawaitable(resolved):
    resolved = await resolved
  if not isinstance(resolved, AutonomousCapabilityBinding):
    raise TypeError(
      "autonomous capability binding resolver must return AutonomousCapabilityBinding"
    )
  if (
    resolved.bind.credential_principal == "user"
    and resolved.materialized_credential.handle.actor_id
    != request.owner_user_id
  ):
    raise ValueError(
      "autonomous user credential handle actor does not match the launch owner"
    )
  if request.required_bind is not None and resolved.bind != request.required_bind:
    raise ValueError("resume capability binding does not match the persisted exact bind")
  if resolved.admitted is None and request.required_bind is None and resolved.bind.capability_id != SESSION_DRIVER_CAPABILITY:
    raise ValueError("unadmitted autonomous capability binding requires a session.driver bind")
  return resolved


__all__ = [
  "AutonomousCapabilityBinding",
  "AutonomousCapabilityBindingRequest",
  "AutonomousCapabilityBindingResolver",
  "AutonomousCapabilityBindingSource",
  "AutonomousLaunchAdmissionRefused",
  "resolve_autonomous_capability_binding",
]
