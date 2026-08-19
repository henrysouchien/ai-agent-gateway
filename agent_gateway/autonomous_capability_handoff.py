from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from typing import Awaitable, Callable, Literal, TypeAlias

from .capability_binding import CapabilityBind, RunMode
from .capability_execution import MaterializedCredential


AutonomousCapabilityBindingSource: TypeAlias = Literal["start", "resume", "schedule"]


def _required_text(value: object, *, field_name: str) -> str:
  text = str(value or "").strip()
  if not text:
    raise ValueError(f"{field_name} must be non-empty")
  return text


@dataclass(frozen=True, slots=True)
class AutonomousCapabilityBindingRequest:
  """Trusted server request for one autonomous session-driver binding."""

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
    if self.source not in {"start", "resume", "schedule"}:
      raise ValueError(f"unsupported autonomous capability binding source: {self.source!r}")
    if self.run_mode not in {"autonomous", "cron"}:
      raise ValueError("autonomous capability binding run_mode must be 'autonomous' or 'cron'")
    if self.source == "schedule" and self.run_mode != "cron":
      raise ValueError("scheduled autonomous binding requests require run_mode='cron'")
    if self.source == "start" and self.run_mode != "autonomous":
      raise ValueError("fresh autonomous binding requests require run_mode='autonomous'")
    if self.source == "resume" and self.run_mode not in {"autonomous", "cron"}:
      raise ValueError("resume binding requests require autonomous or cron run mode")
    if self.source == "resume" and self.required_bind is None:
      raise ValueError("resume binding requests require an exact persisted bind")
    if self.source != "resume" and self.required_bind is not None:
      raise ValueError("only resume binding requests may carry an exact persisted bind")
    if self.required_bind is not None:
      if not isinstance(self.required_bind, CapabilityBind):
        raise TypeError("required_bind must be CapabilityBind")
      if self.required_bind.capability_id != "session.driver":
        raise ValueError("resume binding requests require a session.driver bind")
      if self.required_bind.run_mode != self.run_mode:
        raise ValueError("persisted bind run_mode does not match the resume request")


@dataclass(frozen=True, slots=True)
class AutonomousCapabilityBinding:
  """Exact launch binding plus its sole memory-only credential material."""

  bind: CapabilityBind
  materialized_credential: MaterializedCredential = field(
    repr=False,
    compare=False,
  )

  def __post_init__(self) -> None:
    if not isinstance(self.bind, CapabilityBind):
      raise TypeError("autonomous capability binding bind must be CapabilityBind")
    if self.bind.capability_id != "session.driver":
      raise ValueError("autonomous capability binding requires a session.driver bind")
    if self.bind.run_mode not in {"autonomous", "cron"}:
      raise ValueError(
        "autonomous capability binding requires autonomous or cron run mode"
      )
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
  if resolved.bind.run_mode != request.run_mode:
    raise ValueError("autonomous capability binding run_mode does not match the launch request")
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
  return resolved


__all__ = [
  "AutonomousCapabilityBinding",
  "AutonomousCapabilityBindingRequest",
  "AutonomousCapabilityBindingResolver",
  "AutonomousCapabilityBindingSource",
  "resolve_autonomous_capability_binding",
]
