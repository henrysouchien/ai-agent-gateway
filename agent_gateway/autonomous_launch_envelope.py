from __future__ import annotations

import hashlib
import hmac
import json
import math
import re
import secrets
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Mapping

from .capability_binding import CapabilityBind, CredentialHandle
from .agent_session_log_layout import (
  AutonomousSessionLogAuthority,
  SESSION_LOG_LAYOUT_V2,
  derive_v2_agent_session_log_paths,
)
from .role_validation import require_exact_role
from .session import GatewaySession


AUTONOMOUS_CAPABILITY_ENVELOPE_ENV = (
  "AGENT_AUTONOMOUS_CAPABILITY_ENVELOPE"
)
AUTONOMOUS_TASK_ID_ENV = "AGENT_AUTONOMOUS_TASK_ID"
AUTONOMOUS_CAPABILITY_ENVELOPE_AUDIENCE = (
  "agent-gateway.autonomous-capability/v5"
)
AUTONOMOUS_CAPABILITY_ENVELOPE_VERSION = 5
AUTONOMOUS_CAPABILITY_ENVELOPE_TTL_SECONDS = 60
AUTONOMOUS_CAPABILITY_ENVELOPE_MAX_TTL_SECONDS = 300
AUTONOMOUS_CAPABILITY_ENVELOPE_CLOCK_SKEW_SECONDS = 5
AUTONOMOUS_CAPABILITY_ENVELOPE_MAX_BYTES = 256 * 1024
AUTONOMOUS_CAPABILITY_ENVELOPE_HMAC_MIN_BYTES = 32
AUTONOMOUS_RUNTIME_SESSION_PURPOSE = "autonomous_runtime"

_NONCE_RE = re.compile(r"^[0-9a-f]{32}$")
_SIGNATURE_RE = re.compile(r"^[0-9a-f]{64}$")
_CHANNEL_ID_RE = re.compile(r"^[0-9a-f]{64}$")
_ENVELOPE_FIELDS = frozenset({
  "audience",
  "version",
  "task_id",
  "control_run_id",
  "owner_user_id",
  "channel_id",
  "iat_ns",
  "exp_ns",
  "nonce",
  "capability_bind",
  "workload",
  "control_authority",
  "session_authority",
  "signature",
})
_WORKLOAD_FIELDS = frozenset({
  "profile",
  "mode",
  "task",
  "skill",
  "pack",
  "context",
  "ticker",
  "dev_mode",
  "max_budget_usd",
  "deliver",
  "session_log_authority",
})
_CONTROL_AUTHORITY_FIELDS = frozenset({
  "control_mode",
  "admission_ledger_path",
  "admission_ledger_device",
  "admission_ledger_inode",
  "operator_inbox_path",
  "operator_inbox_device",
  "operator_inbox_inode",
  "approval_decisions_path",
  "approval_decisions_device",
  "approval_decisions_inode",
  "approval_store_path",
  "approval_store_device",
  "approval_store_inode",
})
_SESSION_AUTHORITY_FIELDS = frozenset({
  "ordinary_authority",
  "dispatch_scope",
})
_ORDINARY_AUTHORITY_FIELDS = frozenset({
  "session_id",
  "tenant_id",
  "user_id",
  "owner_user_id",
  "created_at",
  "expires_at",
  "user_email",
  "risk_user_id",
  "role",
  "kind",
  "channel",
  "purpose",
  "raw_user_id",
  "user_slug",
  "user_aliases",
  "identity_status",
  "schema_version",
  "is_public",
  "allow_service_for_interactive",
  "auth_provider",
  "credential_handle",
})
_DISPATCH_SCOPE_FIELDS = frozenset({
  "kind",
  "source",
  "portfolio_name",
  "portfolio_id",
  "display_name",
})


def _closed_mapping(
  value: object,
  *,
  field_name: str,
  fields: frozenset[str],
) -> dict[str, Any]:
  if not isinstance(value, Mapping):
    raise ValueError(
      f"autonomous capability envelope {field_name} must be an object"
    )
  payload = dict(value)
  if set(payload) != fields:
    raise ValueError(
      f"autonomous capability envelope {field_name} violates its closed contract"
    )
  return payload


def _optional_text(
  value: object,
  *,
  field_name: str,
) -> str | None:
  if value is None:
    return None
  return _required_text(value, field_name=field_name)


def _exact_nonnegative_int(
  value: object,
  *,
  field_name: str,
) -> int:
  if isinstance(value, bool) or not isinstance(value, int) or value < 0:
    raise ValueError(
      f"autonomous capability envelope {field_name} must be a nonnegative integer"
    )
  return value


def _exact_positive_int(
  value: object,
  *,
  field_name: str,
) -> int:
  if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
    raise ValueError(
      f"autonomous capability envelope {field_name} must be a positive integer"
    )
  return value


def _canonical_workload_text(
  value: object,
  *,
  field_name: str,
  optional: bool = False,
  free_text: bool = False,
) -> str | None:
  if value is None and optional:
    return None
  if type(value) is not str or not value:
    qualifier = "null or " if optional else ""
    raise ValueError(
      "autonomous launch workload "
      f"{field_name} must be {qualifier}a non-empty string"
    )
  if value != value.strip():
    raise ValueError(
      f"autonomous launch workload {field_name} must be canonical"
    )
  max_bytes = 64 * 1024 if free_text else 512
  if len(value.encode("utf-8")) > max_bytes or "\x00" in value:
    raise ValueError(
      f"autonomous launch workload {field_name} is invalid"
    )
  if (
    not free_text
    and any(ord(character) < 0x20 for character in value)
  ):
    raise ValueError(
      f"autonomous launch workload {field_name} is invalid"
    )
  return value


def _workload_budget(value: object) -> float | None:
  if value is None:
    return None
  if (
    type(value) is not float
  ):
    raise ValueError(
      "autonomous launch workload max_budget_usd "
      "must be null or a finite positive number"
    )
  budget = float(value)
  if not math.isfinite(budget) or budget <= 0:
    raise ValueError(
      "autonomous launch workload max_budget_usd "
      "must be null or a finite positive number"
    )
  return budget


def _canonical_control_path(
  value: object,
  *,
  field_name: str,
  optional: bool = False,
) -> str | None:
  if value is None and optional:
    return None
  if (
    type(value) is not str
    or not value
    or value != value.strip()
    or len(value) > 4096
    or any(ord(character) < 0x20 for character in value)
  ):
    qualifier = "null or " if optional else ""
    raise ValueError(
      "autonomous control authority "
      f"{field_name} must be {qualifier}a canonical absolute path"
    )
  path = Path(value)
  if (
    not path.is_absolute()
    or str(path) != value
    or ".." in path.parts
  ):
    raise ValueError(
      "autonomous control authority "
      f"{field_name} must be a canonical absolute path"
    )
  return value


def _credential_handle_receipt(
  handle: CredentialHandle | None,
) -> dict[str, str | None] | None:
  if handle is None:
    return None
  if type(handle) is not CredentialHandle:
    raise TypeError(
      "autonomous session authority credential_handle must be CredentialHandle"
    )
  return {
    "handle_id": handle.handle_id,
    "provider": handle.provider,
    "principal": handle.principal,
    "tenant_id": handle.tenant_id,
    "actor_id": handle.actor_id,
  }


def _credential_handle_from_receipt(
  value: object,
) -> CredentialHandle | None:
  if value is None:
    return None
  payload = _closed_mapping(
    value,
    field_name="credential_handle",
    fields=frozenset({
      "handle_id",
      "provider",
      "principal",
      "tenant_id",
      "actor_id",
    }),
  )
  try:
    return CredentialHandle(
      handle_id=payload["handle_id"],
      provider=payload["provider"],
      principal=payload["principal"],
      tenant_id=payload["tenant_id"],
      actor_id=payload["actor_id"],
    )
  except (TypeError, ValueError) as exc:
    raise ValueError(
      "autonomous session authority credential_handle is invalid"
    ) from exc


@dataclass(frozen=True, slots=True)
class AutonomousDispatchScope:
  """Closed, immutable portfolio scope carried only inside signed authority."""

  kind: str
  source: str
  portfolio_name: str
  portfolio_id: str | None
  display_name: str | None

  def __post_init__(self) -> None:
    if self.kind != "portfolio":
      raise ValueError(
        "autonomous dispatch scope kind must be 'portfolio'"
      )
    if self.source not in {"active_default", "user_selected"}:
      raise ValueError(
        "autonomous dispatch scope source is invalid"
      )
    portfolio_name = _required_text(
      self.portfolio_name,
      field_name="dispatch_scope.portfolio_name",
    )
    if len(portfolio_name) > 256:
      raise ValueError(
        "autonomous dispatch scope portfolio_name exceeds 256 characters"
      )
    portfolio_id = _optional_text(
      self.portfolio_id,
      field_name="dispatch_scope.portfolio_id",
    )
    display_name = _optional_text(
      self.display_name,
      field_name="dispatch_scope.display_name",
    )
    if portfolio_id is not None and len(portfolio_id) > 256:
      raise ValueError(
        "autonomous dispatch scope portfolio_id exceeds 256 characters"
      )
    if display_name is not None and len(display_name) > 256:
      raise ValueError(
        "autonomous dispatch scope display_name exceeds 256 characters"
      )
    object.__setattr__(self, "portfolio_name", portfolio_name)
    object.__setattr__(self, "portfolio_id", portfolio_id)
    object.__setattr__(self, "display_name", display_name)

  @classmethod
  def from_receipt(
    cls,
    value: object,
  ) -> "AutonomousDispatchScope":
    payload = _closed_mapping(
      value,
      field_name="dispatch_scope",
      fields=_DISPATCH_SCOPE_FIELDS,
    )
    return cls(**payload)

  @classmethod
  def from_mapping(
    cls,
    value: Mapping[str, Any],
  ) -> "AutonomousDispatchScope":
    return cls.from_receipt(value)

  def receipt(self) -> dict[str, str | None]:
    return {
      "kind": self.kind,
      "source": self.source,
      "portfolio_name": self.portfolio_name,
      "portfolio_id": self.portfolio_id,
      "display_name": self.display_name,
    }


@dataclass(frozen=True, slots=True)
class AutonomousLaunchWorkload:
  """Closed executable workload authorized by the launch envelope."""

  profile: str
  mode: Literal["run_once", "task", "skill", "pack"]
  task: str | None
  skill: str | None
  pack: str | None
  context: str | None
  ticker: str | None
  dev_mode: bool
  max_budget_usd: float | None
  deliver: bool
  # This is not an executable CLI argument, so the entrypoint's workload
  # comparison intentionally ignores it.  It is nevertheless mandatory in
  # every signed v5 receipt and is verified independently before bootstrap.
  session_log_authority: AutonomousSessionLogAuthority | None = field(
    default=None,
    compare=False,
  )

  def __post_init__(self) -> None:
    profile = _canonical_workload_text(
      self.profile,
      field_name="profile",
    )
    if profile != profile.lower():
      raise ValueError(
        "autonomous launch workload profile must be lowercase"
      )
    if (
      type(self.mode) is not str
      or self.mode not in {"run_once", "task", "skill", "pack"}
    ):
      raise ValueError(
        "autonomous launch workload mode is invalid"
      )
    if type(self.dev_mode) is not bool:
      raise ValueError(
        "autonomous launch workload dev_mode must be a boolean"
      )
    if type(self.deliver) is not bool:
      raise ValueError(
        "autonomous launch workload deliver must be a boolean"
      )
    if (
      self.session_log_authority is not None
      and type(self.session_log_authority)
      is not AutonomousSessionLogAuthority
    ):
      raise TypeError(
        "autonomous launch workload session_log_authority must be exact"
      )

    task = _canonical_workload_text(
      self.task,
      field_name="task",
      optional=True,
      free_text=True,
    )
    skill = _canonical_workload_text(
      self.skill,
      field_name="skill",
      optional=True,
    )
    pack = _canonical_workload_text(
      self.pack,
      field_name="pack",
      optional=True,
    )
    context = _canonical_workload_text(
      self.context,
      field_name="context",
      optional=True,
      free_text=True,
    )
    ticker = _canonical_workload_text(
      self.ticker,
      field_name="ticker",
      optional=True,
    )
    budget = _workload_budget(self.max_budget_usd)

    if self.mode == "run_once":
      invalid = (
        task is not None
        or skill is not None
        or pack is not None
        or context is not None
        or ticker is not None
        or self.dev_mode
        or budget is not None
        or not self.deliver
      )
    elif self.mode == "task":
      invalid = (
        task is None
        or skill is not None
        or pack is not None
        or context is not None
        or ticker is not None
        or self.dev_mode
        or budget is not None
        or not self.deliver
      )
    elif self.mode == "pack":
      invalid = (
        task is not None
        or skill is not None
        or pack is None
        or context is not None
        or ticker is not None
        or self.dev_mode
        or budget is not None
        or not self.deliver
      )
    else:
      invalid = (
        task is not None
        or skill is None
        or pack is not None
      )
    if invalid:
      raise ValueError(
        "autonomous launch workload fields are incompatible with its mode"
      )

    object.__setattr__(self, "profile", profile)
    object.__setattr__(self, "task", task)
    object.__setattr__(self, "skill", skill)
    object.__setattr__(self, "pack", pack)
    object.__setattr__(self, "context", context)
    object.__setattr__(self, "ticker", ticker)
    object.__setattr__(self, "max_budget_usd", budget)

  @classmethod
  def from_receipt(
    cls,
    value: object,
  ) -> "AutonomousLaunchWorkload":
    payload = _closed_mapping(
      value,
      field_name="workload",
      fields=_WORKLOAD_FIELDS,
    )
    try:
      session_log_authority = AutonomousSessionLogAuthority.from_receipt(
        payload["session_log_authority"]
      )
    except (TypeError, ValueError) as exc:
      raise ValueError(
        "autonomous launch workload session_log_authority is invalid"
      ) from exc
    return cls(
      **{
        **payload,
        "session_log_authority": session_log_authority,
      }
    )

  def receipt(self) -> dict[str, Any]:
    return {
      "profile": self.profile,
      "mode": self.mode,
      "task": self.task,
      "skill": self.skill,
      "pack": self.pack,
      "context": self.context,
      "ticker": self.ticker,
      "dev_mode": self.dev_mode,
      "max_budget_usd": self.max_budget_usd,
      "deliver": self.deliver,
      "session_log_authority": (
        self.session_log_authority.receipt()
        if type(self.session_log_authority)
        is AutonomousSessionLogAuthority
        else None
      ),
    }


@dataclass(frozen=True, slots=True)
class AutonomousControlAuthority:
  """Control transport for one autonomous or same-process run."""

  control_mode: Literal["file", "projected", "memory"]
  admission_ledger_path: str | None
  admission_ledger_device: int | None
  admission_ledger_inode: int | None
  operator_inbox_path: str | None
  operator_inbox_device: int | None
  operator_inbox_inode: int | None
  approval_decisions_path: str | None
  approval_decisions_device: int | None
  approval_decisions_inode: int | None
  approval_store_path: str | None
  approval_store_device: int | None
  approval_store_inode: int | None

  def __post_init__(self) -> None:
    if (
      type(self.control_mode) is not str
      or self.control_mode not in {"file", "projected", "memory"}
    ):
      raise ValueError(
        "autonomous control authority control_mode is invalid"
      )
    ledger_path = _canonical_control_path(
      self.admission_ledger_path,
      field_name="admission_ledger_path",
      optional=True,
    )
    operator_path = _canonical_control_path(
      self.operator_inbox_path,
      field_name="operator_inbox_path",
      optional=True,
    )
    approval_path = _canonical_control_path(
      self.approval_decisions_path,
      field_name="approval_decisions_path",
      optional=True,
    )
    store_path = _canonical_control_path(
      self.approval_store_path,
      field_name="approval_store_path",
      optional=True,
    )

    def normalize_file_identity(
      *,
      path: str | None,
      device: object,
      inode: object,
      field_name: str,
      required: bool,
    ) -> tuple[int | None, int | None]:
      if path is None:
        if device is not None or inode is not None or required:
          raise ValueError(
            "autonomous control authority "
            f"{field_name} path/device/inode must be all present "
            "or all null"
          )
        return None, None
      if (
        isinstance(device, bool)
        or not isinstance(device, int)
        or device < 0
        or isinstance(inode, bool)
        or not isinstance(inode, int)
        or inode <= 0
      ):
        raise ValueError(
          "autonomous control authority "
          f"{field_name} file identity is invalid"
        )
      return device, inode

    file_mode = self.control_mode == "file"
    ledger_device, ledger_inode = normalize_file_identity(
      path=ledger_path,
      device=self.admission_ledger_device,
      inode=self.admission_ledger_inode,
      field_name="admission_ledger",
      required=file_mode,
    )
    operator_device, operator_inode = normalize_file_identity(
      path=operator_path,
      device=self.operator_inbox_device,
      inode=self.operator_inbox_inode,
      field_name="operator_inbox",
      required=file_mode,
    )
    approval_device, approval_inode = normalize_file_identity(
      path=approval_path,
      device=self.approval_decisions_device,
      inode=self.approval_decisions_inode,
      field_name="approval_decisions",
      required=False,
    )
    store_device, store_inode = normalize_file_identity(
      path=store_path,
      device=self.approval_store_device,
      inode=self.approval_store_inode,
      field_name="approval_store",
      required=False,
    )
    if (approval_path is None) != (store_path is None):
      raise ValueError(
        "autonomous control authority approval decision and "
        "store endpoints must be both present or both null"
      )
    if not file_mode and any(
      value is not None
      for value in (
        ledger_path,
        operator_path,
        approval_path,
        store_path,
      )
    ):
      raise ValueError(
        "autonomous in-process control authority forbids file endpoints"
      )
    paths = [
      path
      for path in (
        ledger_path,
        operator_path,
        approval_path,
        store_path,
      )
      if path is not None
    ]
    if len(paths) != len(set(paths)):
      raise ValueError(
        "autonomous control authority paths must be distinct"
      )
    object.__setattr__(
      self,
      "admission_ledger_path",
      ledger_path,
    )
    object.__setattr__(
      self,
      "admission_ledger_device",
      ledger_device,
    )
    object.__setattr__(
      self,
      "admission_ledger_inode",
      ledger_inode,
    )
    object.__setattr__(
      self,
      "operator_inbox_path",
      operator_path,
    )
    object.__setattr__(
      self,
      "operator_inbox_device",
      operator_device,
    )
    object.__setattr__(
      self,
      "operator_inbox_inode",
      operator_inode,
    )
    object.__setattr__(
      self,
      "approval_decisions_path",
      approval_path,
    )
    object.__setattr__(
      self,
      "approval_decisions_device",
      approval_device,
    )
    object.__setattr__(
      self,
      "approval_decisions_inode",
      approval_inode,
    )
    object.__setattr__(
      self,
      "approval_store_path",
      store_path,
    )
    object.__setattr__(
      self,
      "approval_store_device",
      store_device,
    )
    object.__setattr__(
      self,
      "approval_store_inode",
      store_inode,
    )

  @classmethod
  def from_receipt(
    cls,
    value: object,
  ) -> "AutonomousControlAuthority":
    payload = _closed_mapping(
      value,
      field_name="control_authority",
      fields=_CONTROL_AUTHORITY_FIELDS,
    )
    return cls(**payload)

  def receipt(self) -> dict[str, str | int | None]:
    return {
      "control_mode": self.control_mode,
      "admission_ledger_path": self.admission_ledger_path,
      "admission_ledger_device": self.admission_ledger_device,
      "admission_ledger_inode": self.admission_ledger_inode,
      "operator_inbox_path": self.operator_inbox_path,
      "operator_inbox_device": self.operator_inbox_device,
      "operator_inbox_inode": self.operator_inbox_inode,
      "approval_decisions_path": self.approval_decisions_path,
      "approval_decisions_device": self.approval_decisions_device,
      "approval_decisions_inode": self.approval_decisions_inode,
      "approval_store_path": self.approval_store_path,
      "approval_store_device": self.approval_store_device,
      "approval_store_inode": self.approval_store_inode,
    }


@dataclass(frozen=True, slots=True)
class OrdinaryAutonomousSessionAuthority:
  """Secret-free server authority for one ordinary child GatewaySession."""

  session_id: str
  tenant_id: str
  user_id: str
  owner_user_id: str
  created_at: int
  expires_at: int
  user_email: str | None
  risk_user_id: int
  role: str
  kind: str
  channel: str
  purpose: str
  raw_user_id: str | None
  user_slug: str | None
  user_aliases: tuple[str, ...]
  identity_status: str | None
  schema_version: int
  is_public: bool
  allow_service_for_interactive: bool
  auth_provider: str
  credential_handle: CredentialHandle | None

  def __post_init__(self) -> None:
    for field_name in (
      "session_id",
      "tenant_id",
      "user_id",
      "owner_user_id",
      "channel",
      "auth_provider",
    ):
      object.__setattr__(
        self,
        field_name,
        _required_text(
          getattr(self, field_name),
          field_name=f"session_authority.{field_name}",
        ),
      )
    if self.user_id != self.owner_user_id:
      raise ValueError(
        "ordinary autonomous session authority requires its owner actor"
      )
    require_exact_role(self.role)
    if self.kind != "chat":
      raise ValueError(
        "ordinary autonomous session authority requires chat identity"
      )
    if self.purpose != AUTONOMOUS_RUNTIME_SESSION_PURPOSE:
      raise ValueError(
        "ordinary autonomous session authority purpose is invalid"
      )
    created_at = _exact_nonnegative_int(
      self.created_at,
      field_name="session_authority.created_at",
    )
    expires_at = _exact_positive_int(
      self.expires_at,
      field_name="session_authority.expires_at",
    )
    if expires_at <= created_at:
      raise ValueError(
        "ordinary autonomous session authority expiry must follow creation"
      )
    _exact_nonnegative_int(
      self.risk_user_id,
      field_name="session_authority.risk_user_id",
    )
    _exact_positive_int(
      self.schema_version,
      field_name="session_authority.schema_version",
    )
    if (
      type(self.is_public) is not bool
      or self.is_public
      or type(self.allow_service_for_interactive) is not bool
    ):
      raise ValueError(
        "ordinary autonomous session authority violates its fixed policy"
      )
    user_email = _optional_text(
      self.user_email,
      field_name="session_authority.user_email",
    )
    raw_user_id = _optional_text(
      self.raw_user_id,
      field_name="session_authority.raw_user_id",
    )
    user_slug = _optional_text(
      self.user_slug,
      field_name="session_authority.user_slug",
    )
    identity_status = _optional_text(
      self.identity_status,
      field_name="session_authority.identity_status",
    )
    if type(self.user_aliases) is not tuple or len(self.user_aliases) > 64:
      raise ValueError(
        "ordinary autonomous session authority aliases are invalid"
      )
    aliases = tuple(
      _required_text(
        alias,
        field_name="session_authority.user_alias",
      )
      for alias in self.user_aliases
    )
    if len(aliases) != len(set(aliases)):
      raise ValueError(
        "ordinary autonomous session authority aliases contain duplicates"
      )
    handle = self.credential_handle
    if handle is not None:
      if (
        type(handle) is not CredentialHandle
        or handle.tenant_id != self.tenant_id
        or handle.provider != self.auth_provider
        or (
          handle.principal == "user"
          and handle.actor_id != self.owner_user_id
        )
      ):
        raise ValueError(
          "ordinary autonomous session credential authority is invalid"
        )
    object.__setattr__(self, "user_email", user_email)
    object.__setattr__(self, "raw_user_id", raw_user_id)
    object.__setattr__(self, "user_slug", user_slug)
    object.__setattr__(self, "user_aliases", aliases)
    object.__setattr__(self, "identity_status", identity_status)
    object.__setattr__(
      self,
      "auth_provider",
      self.auth_provider.lower(),
    )

  @classmethod
  def from_receipt(
    cls,
    value: object,
  ) -> "OrdinaryAutonomousSessionAuthority":
    payload = _closed_mapping(
      value,
      field_name="ordinary_session_authority",
      fields=_ORDINARY_AUTHORITY_FIELDS,
    )
    aliases = payload["user_aliases"]
    if not isinstance(aliases, list):
      raise ValueError(
        "ordinary autonomous session authority aliases must be a JSON array"
      )
    return cls(
      **{
        **payload,
        "user_aliases": tuple(aliases),
        "credential_handle": _credential_handle_from_receipt(
          payload["credential_handle"]
        ),
      }
    )

  def receipt(self) -> dict[str, Any]:
    return {
      "session_id": self.session_id,
      "tenant_id": self.tenant_id,
      "user_id": self.user_id,
      "owner_user_id": self.owner_user_id,
      "created_at": self.created_at,
      "expires_at": self.expires_at,
      "user_email": self.user_email,
      "risk_user_id": self.risk_user_id,
      "role": self.role,
      "kind": self.kind,
      "channel": self.channel,
      "purpose": self.purpose,
      "raw_user_id": self.raw_user_id,
      "user_slug": self.user_slug,
      "user_aliases": list(self.user_aliases),
      "identity_status": self.identity_status,
      "schema_version": self.schema_version,
      "is_public": self.is_public,
      "allow_service_for_interactive": (
        self.allow_service_for_interactive
      ),
      "auth_provider": self.auth_provider,
      "credential_handle": _credential_handle_receipt(
        self.credential_handle
      ),
    }


@dataclass(frozen=True, slots=True)
class AutonomousSessionAuthority:
  """Exact ordinary authority for one autonomous child session."""

  ordinary_authority: OrdinaryAutonomousSessionAuthority
  dispatch_scope: AutonomousDispatchScope | None

  def __post_init__(self) -> None:
    if (
      type(self.ordinary_authority)
      is not OrdinaryAutonomousSessionAuthority
    ):
      raise ValueError(
        "ordinary autonomous session authority is invalid"
      )

  @classmethod
  def ordinary(
    cls,
    authority: OrdinaryAutonomousSessionAuthority,
    *,
    dispatch_scope: AutonomousDispatchScope | None = None,
  ) -> "AutonomousSessionAuthority":
    return cls(
      ordinary_authority=authority,
      dispatch_scope=dispatch_scope,
    )

  @classmethod
  def from_receipt(
    cls,
    value: object,
  ) -> "AutonomousSessionAuthority":
    payload = _closed_mapping(
      value,
      field_name="session_authority",
      fields=_SESSION_AUTHORITY_FIELDS,
    )
    return cls.ordinary(
      OrdinaryAutonomousSessionAuthority.from_receipt(
        payload["ordinary_authority"]
      ),
      dispatch_scope=(
        AutonomousDispatchScope.from_receipt(
          payload["dispatch_scope"]
        )
        if payload["dispatch_scope"] is not None
        else None
      ),
    )

  def receipt(self) -> dict[str, Any]:
    return {
      "ordinary_authority": self.ordinary_authority.receipt(),
      "dispatch_scope": (
        self.dispatch_scope.receipt()
        if self.dispatch_scope is not None
        else None
      ),
    }

  def to_gateway_session(self) -> GatewaySession:
    authority = self.ordinary_authority
    if type(authority) is not OrdinaryAutonomousSessionAuthority:
      raise RuntimeError(
        "ordinary autonomous session authority is unavailable"
      )
    session = GatewaySession(
      session_id=authority.session_id,
      api_key_hash="",
      created_at=authority.created_at,
      expires_at=authority.expires_at,
      user_id=authority.user_id,
      user_email=authority.user_email,
      risk_user_id=authority.risk_user_id,
      role=authority.role,
      kind=authority.kind,
      auth_config={"provider": authority.auth_provider},
      tenant_id=authority.tenant_id,
      session_credential_handle=authority.credential_handle,
      allow_service_for_interactive=(
        authority.allow_service_for_interactive
      ),
      channel=authority.channel,
      owner_user_id=authority.owner_user_id,
      raw_user_id=authority.raw_user_id,
      user_slug=authority.user_slug,
      user_aliases=authority.user_aliases,
      identity_status=authority.identity_status,
      is_public=authority.is_public,
      schema_version=authority.schema_version,
      purpose=authority.purpose,
    )
    session.dispatch_scope = (
      self.dispatch_scope.receipt()
      if self.dispatch_scope is not None
      else None
    )
    return session


@dataclass(frozen=True, slots=True)
class AutonomousLaunchEnvelope:
  """Verified, secret-free autonomous launch authorization."""

  audience: str
  version: int
  task_id: str
  control_run_id: str
  owner_user_id: str
  channel_id: str
  iat_ns: int
  exp_ns: int
  nonce: str
  bind: CapabilityBind
  workload: AutonomousLaunchWorkload
  control_authority: AutonomousControlAuthority
  session_authority: AutonomousSessionAuthority
  signature: str

  def payload(self) -> dict[str, Any]:
    return {
      "audience": self.audience,
      "version": self.version,
      "task_id": self.task_id,
      "control_run_id": self.control_run_id,
      "owner_user_id": self.owner_user_id,
      "channel_id": self.channel_id,
      "iat_ns": self.iat_ns,
      "exp_ns": self.exp_ns,
      "nonce": self.nonce,
      "capability_bind": self.bind.receipt(),
      "workload": self.workload.receipt(),
      "control_authority": self.control_authority.receipt(),
      "session_authority": self.session_authority.receipt(),
      "signature": self.signature,
    }


def _canonical_json(value: Any) -> str:
  return json.dumps(
    value,
    sort_keys=True,
    separators=(",", ":"),
    ensure_ascii=False,
    allow_nan=False,
  )


def _hmac_key(secret: bytes | str) -> bytes:
  if isinstance(secret, bytes):
    key = secret
  elif isinstance(secret, str):
    key = secret.encode("utf-8")
  else:
    raise TypeError(
      "autonomous capability envelope HMAC secret must be bytes or str"
    )
  if len(key) < AUTONOMOUS_CAPABILITY_ENVELOPE_HMAC_MIN_BYTES:
    raise ValueError(
      "autonomous capability envelope HMAC secret "
      f"must be at least {AUTONOMOUS_CAPABILITY_ENVELOPE_HMAC_MIN_BYTES} bytes"
    )
  return key


def _required_text(value: object, *, field_name: str) -> str:
  if type(value) is not str or not value.strip():
    raise ValueError(
      "autonomous capability envelope "
      f"{field_name} must be a non-empty string"
    )
  normalized = value.strip()
  if len(normalized) > 512 or any(
    ord(character) < 0x20
    for character in normalized
  ):
    raise ValueError(
      f"autonomous capability envelope {field_name} is invalid"
    )
  return normalized


def _autonomous_bind(bind: object) -> CapabilityBind:
  if type(bind) is not CapabilityBind:
    raise TypeError(
      "autonomous capability envelope bind must be CapabilityBind"
    )
  if bind.capability_id != "session.driver":
    raise ValueError(
      "autonomous capability envelope requires a session.driver bind"
    )
  if bind.run_mode not in {"autonomous", "cron"}:
    raise ValueError(
      "autonomous capability envelope bind must use autonomous or cron run mode"
    )
  return bind


def _bind_sha256(bind: CapabilityBind) -> str:
  return hashlib.sha256(
    _canonical_json(bind.receipt()).encode("utf-8")
  ).hexdigest()


def _closed_object(
  pairs: list[tuple[str, Any]],
) -> dict[str, Any]:
  payload: dict[str, Any] = {}
  for key, value in pairs:
    if key in payload:
      raise ValueError(
        "autonomous capability envelope "
        f"contains duplicate field: {key}"
      )
    payload[key] = value
  return payload


def _signature(
  secret: bytes | str,
  unsigned_payload: dict[str, Any],
) -> str:
  return hmac.new(
    _hmac_key(secret),
    _canonical_json(unsigned_payload).encode("utf-8"),
    hashlib.sha256,
  ).hexdigest()


def _timestamp_ns(value: object, *, field_name: str) -> int:
  if (
    isinstance(value, bool)
    or not isinstance(value, int)
    or value <= 0
  ):
    raise ValueError(
      "autonomous capability envelope "
      f"{field_name} must be a positive integer"
    )
  return value


def _validate_session_authority_bindings(
  session_authority: AutonomousSessionAuthority,
  *,
  task_id: str,
  owner_user_id: str,
  bind: CapabilityBind,
  issued_at_ns: int,
  expires_at_ns: int,
  current_time_ns: int,
) -> None:
  if type(session_authority) is not AutonomousSessionAuthority:
    raise TypeError(
      "autonomous capability envelope session_authority "
      "must be AutonomousSessionAuthority"
    )
  ordinary = session_authority.ordinary_authority
  if (
    type(ordinary) is not OrdinaryAutonomousSessionAuthority
    or ordinary.session_id != task_id
    or ordinary.user_id != owner_user_id
    or ordinary.owner_user_id != owner_user_id
    or ordinary.auth_provider != bind.provider
  ):
    raise ValueError(
      "ordinary autonomous session authority bindings do not match"
    )
  handle = ordinary.credential_handle
  if (
    type(handle) is not CredentialHandle
    or handle.handle_id != bind.credential_ref
    or handle.principal != bind.credential_principal
    or handle.provider != bind.provider
    or (
      handle.principal == "user"
      and handle.actor_id != owner_user_id
    )
    or (
      handle.principal == "service"
      and handle.actor_id is not None
    )
  ):
    raise ValueError(
      "ordinary autonomous session credential authority does not match"
    )
  if (
    ordinary.created_at * 1_000_000_000 > issued_at_ns
    or ordinary.expires_at * 1_000_000_000 < expires_at_ns
    or ordinary.created_at * 1_000_000_000 > current_time_ns
    or ordinary.expires_at * 1_000_000_000 <= current_time_ns
  ):
    raise ValueError(
      "ordinary autonomous session authority lifetime "
      "does not cover launch admission"
    )


def _validate_session_log_authority_bindings(
  workload: AutonomousLaunchWorkload,
  *,
  session_authority: AutonomousSessionAuthority,
  owner_user_id: str,
  bind: CapabilityBind,
) -> None:
  authority = workload.session_log_authority
  if type(authority) is not AutonomousSessionLogAuthority:
    raise TypeError(
      "autonomous capability envelope workload requires exact "
      "session-log authority"
    )
  ordinary = session_authority.ordinary_authority
  if type(ordinary) is not OrdinaryAutonomousSessionAuthority:
    raise TypeError(
      "session-log authority requires exact ordinary session authority"
    )
  try:
    root, active, meta, digest = derive_v2_agent_session_log_paths(
      authority.base_path,
      tenant=ordinary.tenant_id,
      owner=owner_user_id,
      workload_profile=workload.profile,
      provider=bind.provider,
      provider_session_epoch=authority.provider_session_epoch,
    )
  except (TypeError, ValueError) as exc:
    raise ValueError(
      "session-log authority identity is invalid"
    ) from exc
  if authority.layout != SESSION_LOG_LAYOUT_V2:
    return
  if (
    authority.root_path != str(root)
    or authority.active_path != str(active)
    or authority.meta_path != str(meta)
    or authority.storage_identity_digest != digest
  ):
    raise ValueError(
      "session-log authority does not match authenticated workload identity"
    )


def sign_autonomous_launch_envelope(
  secret: bytes | str,
  *,
  task_id: str,
  control_run_id: str,
  owner_user_id: str,
  channel_id: str,
  bind: CapabilityBind,
  workload: AutonomousLaunchWorkload,
  control_authority: AutonomousControlAuthority,
  session_authority: AutonomousSessionAuthority,
  ttl_seconds: int = AUTONOMOUS_CAPABILITY_ENVELOPE_TTL_SECONDS,
  now_ns: int | None = None,
  nonce: str | None = None,
) -> str:
  """Return the sole canonical signed launch-envelope contract."""

  bind = _autonomous_bind(bind)
  if type(workload) is not AutonomousLaunchWorkload:
    raise TypeError(
      "autonomous capability envelope workload "
      "must be AutonomousLaunchWorkload"
    )
  if type(control_authority) is not AutonomousControlAuthority:
    raise TypeError(
      "autonomous capability envelope control_authority "
      "must be AutonomousControlAuthority"
    )
  if control_authority.control_mode == "memory":
    raise ValueError(
      "in-memory control authority cannot cross a process boundary"
    )
  if isinstance(ttl_seconds, bool) or not isinstance(ttl_seconds, int):
    raise TypeError(
      "autonomous capability envelope ttl_seconds must be an integer"
    )
  if not 1 <= ttl_seconds <= (
    AUTONOMOUS_CAPABILITY_ENVELOPE_MAX_TTL_SECONDS
  ):
    raise ValueError(
      "autonomous capability envelope ttl_seconds must be between 1 and "
      f"{AUTONOMOUS_CAPABILITY_ENVELOPE_MAX_TTL_SECONDS}"
    )
  issued_at_ns = _timestamp_ns(
    time.time_ns() if now_ns is None else now_ns,
    field_name="now_ns",
  )
  expires_at_ns = issued_at_ns + ttl_seconds * 1_000_000_000
  resolved_nonce = (
    secrets.token_hex(16)
    if nonce is None
    else nonce
  )
  if (
    type(resolved_nonce) is not str
    or _NONCE_RE.fullmatch(resolved_nonce) is None
  ):
    raise ValueError(
      "autonomous capability envelope nonce "
      "must be 32 lowercase hex characters"
    )
  normalized_task_id = _required_text(
    task_id,
    field_name="task_id",
  )
  normalized_control_run_id = _required_text(
    control_run_id,
    field_name="control_run_id",
  )
  normalized_owner_user_id = _required_text(
    owner_user_id,
    field_name="owner_user_id",
  )
  normalized_channel_id = _required_text(
    channel_id,
    field_name="channel_id",
  )
  if _CHANNEL_ID_RE.fullmatch(normalized_channel_id) is None:
    raise ValueError(
      "autonomous capability envelope channel_id "
      "must be 64 lowercase hexadecimal characters"
    )
  _validate_session_authority_bindings(
    session_authority,
    task_id=normalized_task_id,
    owner_user_id=normalized_owner_user_id,
    bind=bind,
    issued_at_ns=issued_at_ns,
    expires_at_ns=expires_at_ns,
    current_time_ns=issued_at_ns,
  )
  _validate_session_log_authority_bindings(
    workload,
    session_authority=session_authority,
    owner_user_id=normalized_owner_user_id,
    bind=bind,
  )
  unsigned_payload = {
    "audience": AUTONOMOUS_CAPABILITY_ENVELOPE_AUDIENCE,
    "version": AUTONOMOUS_CAPABILITY_ENVELOPE_VERSION,
    "task_id": normalized_task_id,
    "control_run_id": normalized_control_run_id,
    "owner_user_id": normalized_owner_user_id,
    "channel_id": normalized_channel_id,
    "iat_ns": issued_at_ns,
    "exp_ns": expires_at_ns,
    "nonce": resolved_nonce,
    "capability_bind": bind.receipt(),
    "workload": workload.receipt(),
    "control_authority": control_authority.receipt(),
    "session_authority": session_authority.receipt(),
  }
  envelope_json = _canonical_json({
    **unsigned_payload,
    "signature": _signature(secret, unsigned_payload),
  })
  if (
    len(envelope_json.encode("utf-8"))
    > AUTONOMOUS_CAPABILITY_ENVELOPE_MAX_BYTES
  ):
    raise ValueError(
      "autonomous capability envelope exceeds its byte bound"
    )
  return envelope_json


def _decode_autonomous_launch_envelope(
  secret: bytes | str | None,
  envelope_json: str,
  *,
  authenticate_signature: bool,
  now_ns: int | None = None,
  max_ttl_seconds: int = (
    AUTONOMOUS_CAPABILITY_ENVELOPE_MAX_TTL_SECONDS
  ),
  clock_skew_seconds: int = (
    AUTONOMOUS_CAPABILITY_ENVELOPE_CLOCK_SKEW_SECONDS
  ),
) -> AutonomousLaunchEnvelope:
  """Decode the closed envelope after its caller establishes authenticity."""

  if type(envelope_json) is not str or not envelope_json.strip():
    raise ValueError(
      "autonomous capability envelope "
      "must be a non-empty JSON string"
    )
  if (
    len(envelope_json.encode("utf-8"))
    > AUTONOMOUS_CAPABILITY_ENVELOPE_MAX_BYTES
  ):
    raise ValueError(
      "autonomous capability envelope exceeds its byte bound"
    )
  try:
    raw = json.loads(
      envelope_json,
      object_pairs_hook=_closed_object,
    )
  except json.JSONDecodeError as exc:
    raise ValueError(
      "autonomous capability envelope must be valid JSON"
    ) from exc
  if not isinstance(raw, dict):
    raise ValueError(
      "autonomous capability envelope must be a JSON object"
    )
  try:
    canonical_envelope = _canonical_json(raw)
  except (TypeError, ValueError) as exc:
    raise ValueError(
      "autonomous capability envelope must contain canonical JSON values"
    ) from exc
  if canonical_envelope != envelope_json:
    raise ValueError(
      "autonomous capability envelope must use canonical JSON"
    )
  fields = set(raw)
  missing = _ENVELOPE_FIELDS - fields
  extra = fields - _ENVELOPE_FIELDS
  if missing:
    raise ValueError(
      "autonomous capability envelope is missing fields: "
      + ", ".join(sorted(missing))
    )
  if extra:
    raise ValueError(
      "autonomous capability envelope has unexpected fields: "
      + ", ".join(sorted(extra))
    )
  signature = raw["signature"]
  if (
    type(signature) is not str
    or _SIGNATURE_RE.fullmatch(signature) is None
  ):
    raise ValueError(
      "autonomous capability envelope signature "
      "must be 64 lowercase hex characters"
    )
  unsigned_payload = dict(raw)
  unsigned_payload.pop("signature")
  if authenticate_signature:
    if secret is None:
      raise RuntimeError(
        "autonomous capability envelope verifier lost its secret"
      )
    expected_signature = _signature(secret, unsigned_payload)
    if not hmac.compare_digest(signature, expected_signature):
      raise ValueError(
        "autonomous capability envelope signature is invalid"
      )

  audience = _required_text(raw["audience"], field_name="audience")
  if audience != AUTONOMOUS_CAPABILITY_ENVELOPE_AUDIENCE:
    raise ValueError(
      "autonomous capability envelope audience is invalid"
    )
  version = raw["version"]
  if isinstance(version, bool) or not isinstance(version, int):
    raise ValueError(
      "autonomous capability envelope version must be an integer"
    )
  if version != AUTONOMOUS_CAPABILITY_ENVELOPE_VERSION:
    raise ValueError(
      "autonomous capability envelope version is unsupported"
    )

  task_id = _required_text(raw["task_id"], field_name="task_id")
  control_run_id = _required_text(
    raw["control_run_id"],
    field_name="control_run_id",
  )
  owner_user_id = _required_text(
    raw["owner_user_id"],
    field_name="owner_user_id",
  )
  channel_id = _required_text(
    raw["channel_id"],
    field_name="channel_id",
  )
  if _CHANNEL_ID_RE.fullmatch(channel_id) is None:
    raise ValueError(
      "autonomous capability envelope channel_id "
      "must be 64 lowercase hexadecimal characters"
    )

  issued_at_ns = _timestamp_ns(raw["iat_ns"], field_name="iat_ns")
  expires_at_ns = _timestamp_ns(raw["exp_ns"], field_name="exp_ns")
  if (
    isinstance(max_ttl_seconds, bool)
    or not isinstance(max_ttl_seconds, int)
    or max_ttl_seconds < 1
  ):
    raise ValueError(
      "autonomous capability envelope "
      "max_ttl_seconds must be positive"
    )
  if (
    isinstance(clock_skew_seconds, bool)
    or not isinstance(clock_skew_seconds, int)
    or clock_skew_seconds < 0
  ):
    raise ValueError(
      "autonomous capability envelope "
      "clock_skew_seconds must be non-negative"
    )
  if expires_at_ns <= issued_at_ns:
    raise ValueError(
      "autonomous capability envelope expiry must follow issuance"
    )
  if (
    expires_at_ns - issued_at_ns
    > max_ttl_seconds * 1_000_000_000
  ):
    raise ValueError(
      "autonomous capability envelope TTL exceeds the allowed maximum"
    )
  current_time_ns = _timestamp_ns(
    time.time_ns() if now_ns is None else now_ns,
    field_name="now_ns",
  )
  skew_ns = clock_skew_seconds * 1_000_000_000
  if issued_at_ns > current_time_ns + skew_ns:
    raise ValueError(
      "autonomous capability envelope was issued in the future"
    )
  if expires_at_ns < current_time_ns - skew_ns:
    raise ValueError(
      "autonomous capability envelope has expired"
    )

  nonce = raw["nonce"]
  if (
    type(nonce) is not str
    or _NONCE_RE.fullmatch(nonce) is None
  ):
    raise ValueError(
      "autonomous capability envelope nonce "
      "must be 32 lowercase hex characters"
    )
  try:
    bind = CapabilityBind.from_receipt(raw["capability_bind"])
  except (TypeError, ValueError) as exc:
    raise ValueError(
      f"autonomous capability envelope bind is invalid: {exc}"
    ) from exc
  bind = _autonomous_bind(bind)
  try:
    workload = AutonomousLaunchWorkload.from_receipt(
      raw["workload"]
    )
  except (TypeError, ValueError) as exc:
    raise ValueError(
      "autonomous capability envelope workload is invalid"
    ) from exc
  try:
    control_authority = AutonomousControlAuthority.from_receipt(
      raw["control_authority"]
    )
    if control_authority.control_mode == "memory":
      raise ValueError(
        "in-memory control authority cannot cross a process boundary"
      )
  except (TypeError, ValueError) as exc:
    raise ValueError(
      "autonomous capability envelope control_authority is invalid"
    ) from exc
  try:
    session_authority = AutonomousSessionAuthority.from_receipt(
      raw["session_authority"]
    )
  except (TypeError, ValueError) as exc:
    raise ValueError(
      "autonomous capability envelope session_authority is invalid"
    ) from exc
  _validate_session_authority_bindings(
    session_authority,
    task_id=task_id,
    owner_user_id=owner_user_id,
    bind=bind,
    issued_at_ns=issued_at_ns,
    expires_at_ns=expires_at_ns,
    current_time_ns=current_time_ns,
  )
  _validate_session_log_authority_bindings(
    workload,
    session_authority=session_authority,
    owner_user_id=owner_user_id,
    bind=bind,
  )
  return AutonomousLaunchEnvelope(
    audience=audience,
    version=version,
    task_id=task_id,
    control_run_id=control_run_id,
    owner_user_id=owner_user_id,
    channel_id=channel_id,
    iat_ns=issued_at_ns,
    exp_ns=expires_at_ns,
    nonce=nonce,
    bind=bind,
    workload=workload,
    control_authority=control_authority,
    session_authority=session_authority,
    signature=signature,
  )


def verify_autonomous_launch_envelope(
  secret: bytes | str,
  envelope_json: str,
  *,
  now_ns: int | None = None,
  max_ttl_seconds: int = (
    AUTONOMOUS_CAPABILITY_ENVELOPE_MAX_TTL_SECONDS
  ),
  clock_skew_seconds: int = (
    AUTONOMOUS_CAPABILITY_ENVELOPE_CLOCK_SKEW_SECONDS
  ),
) -> AutonomousLaunchEnvelope:
  """Verify the closed launch envelope before any child side effect."""

  return _decode_autonomous_launch_envelope(
    secret,
    envelope_json,
    authenticate_signature=True,
    now_ns=now_ns,
    max_ttl_seconds=max_ttl_seconds,
    clock_skew_seconds=clock_skew_seconds,
  )


def _decode_broker_verified_autonomous_launch_envelope(
  envelope_json: str,
  *,
  now_ns: int | None = None,
) -> AutonomousLaunchEnvelope:
  """Decode only after the private broker admitted this exact envelope."""

  return _decode_autonomous_launch_envelope(
    None,
    envelope_json,
    authenticate_signature=False,
    now_ns=now_ns,
  )


__all__ = [
  "AUTONOMOUS_CAPABILITY_ENVELOPE_AUDIENCE",
  "AUTONOMOUS_CAPABILITY_ENVELOPE_CLOCK_SKEW_SECONDS",
  "AUTONOMOUS_CAPABILITY_ENVELOPE_ENV",
  "AUTONOMOUS_CAPABILITY_ENVELOPE_MAX_TTL_SECONDS",
  "AUTONOMOUS_CAPABILITY_ENVELOPE_MAX_BYTES",
  "AUTONOMOUS_CAPABILITY_ENVELOPE_HMAC_MIN_BYTES",
  "AUTONOMOUS_CAPABILITY_ENVELOPE_TTL_SECONDS",
  "AUTONOMOUS_CAPABILITY_ENVELOPE_VERSION",
  "AUTONOMOUS_RUNTIME_SESSION_PURPOSE",
  "AUTONOMOUS_TASK_ID_ENV",
  "AutonomousControlAuthority",
  "AutonomousDispatchScope",
  "AutonomousLaunchEnvelope",
  "AutonomousLaunchWorkload",
  "AutonomousSessionAuthority",
  "OrdinaryAutonomousSessionAuthority",
  "sign_autonomous_launch_envelope",
  "verify_autonomous_launch_envelope",
]
