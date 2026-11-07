from __future__ import annotations

import hashlib
import inspect
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, TYPE_CHECKING

from .approval_policy import (
  MONEY_BOUNDARY_TOOL_CLASSES,
  ApprovalConstraint,
  ApprovalDecision,
  ApprovalRequest,
  ApprovalRequestPayload,
  RunContext,
)
from hank_model_transport.policy_imports import (
  load_server_policy_module,
  resolve_effective_role,
  resolve_server_policy_tool_class,
)

from .run_settings import RunSettingsError, load_run_settings, pinned_run_settings

if TYPE_CHECKING:
  from .run_settings import ApprovalPolicyMode


class ApprovalSettingsPinError(RuntimeError):
  """A run carries no launch approval settings, so it cannot start or resume."""


@dataclass(frozen=True)
class ApprovalSettings:
  """The two approval switches one decision reads, from the one owner."""

  approval_policy: ApprovalPolicyMode
  approve_live_orders: bool

  def record(self) -> dict[str, object]:
    """The form a launch or a persisted run carries across a process."""

    return {
      "approval_policy": self.approval_policy,
      "approve_live_orders": self.approve_live_orders,
    }

  @classmethod
  def from_record(cls, record: Mapping[str, object]) -> ApprovalSettings:
    """Parse a carried record; unknown keys, modes and non-booleans are refused."""

    if set(record) != {"approval_policy", "approve_live_orders"}:
      raise RunSettingsError(f"approval settings record keys: {sorted(record)}")
    validated = pinned_run_settings(**record)
    return cls(validated.approval_policy, validated.approve_live_orders)


# Deliberate, not a fallback: a headless batch, scheduled or autonomous run
# binds the operator values at launch (``pin_approval_settings``) and keeps
# them for the run's whole life. The parent owns that snapshot; a child never
# states its own. Unbound means an interactive session, the one path that
# follows the live operator file, so a flip of the setting is in force at its
# next decision.
_pinned: ContextVar[ApprovalSettings | None] = ContextVar(
  "approval_settings_pin", default=None,
)


def current_approval_settings() -> ApprovalSettings:
  """The pinned run values when a run bound them, else the live operator file."""

  pinned = _pinned.get()
  if pinned is not None:
    return pinned
  settings = load_run_settings()
  return ApprovalSettings(settings.approval_policy, settings.approve_live_orders)


@contextmanager
def pin_approval_settings(
  settings: ApprovalSettings | None = None,
) -> Iterator[ApprovalSettings]:
  """Bind the approval switches for one run; ``None`` snapshots the current ones."""

  pinned = settings if settings is not None else current_approval_settings()
  token = _pinned.set(pinned)
  try:
    yield pinned
  finally:
    _pinned.reset(token)


# Registered effects that never need a human on their own: `support` is the
# read-support effect (file/memory/doc/web reads); `preview` is deliberately
# absent because it produces a proposal or artifact, so a careful run asks.
_NO_CARD_CLASSES = frozenset({"read", "pure_transform", "support"})


def approval_requires_human(
  tool_class: str,
  approval_constraint: ApprovalConstraint,
  settings: ApprovalSettings,
) -> bool:
  """The class decision shared by dispatch admission and durable decisions.

  A ``fresh_human_owner`` request keeps its frozen human authority under every
  setting. A live order asks unless ``approve_live_orders`` is on, whatever the
  policy mode; ``ask_all`` also asks for every class outside the read-like ones.
  """

  if approval_constraint == "fresh_human_owner":
    return True
  if tool_class in MONEY_BOUNDARY_TOOL_CLASSES:
    return not settings.approve_live_orders
  return settings.approval_policy == "ask_all" and tool_class not in _NO_CARD_CLASSES


def needs_human_approval(
  tool_class: str,
  approval_constraint: ApprovalConstraint = "standard",
) -> bool:
  """Read the operator settings in one place, then apply the pure class rule."""

  return approval_requires_human(
    tool_class, approval_constraint, current_approval_settings(),
  )


def resolve_approval_class(
  tool_name: str,
  *,
  policy_tool_name: str | None = None,
  runtime_server: str | None = None,
) -> str:
  """The registered class the owner decides on: local effect, else MCP class.

  A name the local registration does not own is an MCP tool; with no bound
  host policy there is no registry to ask and the MCP resolver answers.
  """

  policy_module = load_server_policy_module()
  if policy_module is not None and not runtime_server:
    local_effect = policy_module.get_local_tool_registration_effect(
      policy_tool_name or tool_name,
    )
    if local_effect is not None:
      return local_effect
  return resolve_server_policy_tool_class(
    tool_name,
    policy_tool_name=policy_tool_name,
    runtime_server=runtime_server,
  )


def requires_approval(
  tool_name: str,
  tool_input: dict[str, Any] | None = None,
  qualifier: str = "",
) -> bool:
  """Owner callback for name-based dependency-injection callers."""

  _ = tool_input, qualifier
  return needs_human_approval(resolve_approval_class(tool_name))


class SingleUserApprovalPolicy:
  """The one owner of which classes need a human, for every channel.

  The approve-all default asks only at the money boundary, and
  ``approve_live_orders`` waives even that. Careful runs may select ask-all for
  other writes. Reads and pure transforms, including code execution, need no
  card in any mode; containment belongs to the backend.
  """

  policy_id = "single-user"
  policy_version = "5"

  def __init__(self, *, store: Any | None = None) -> None:
    self._store = store
    try:
      source = inspect.getsource(type(self))
    except Exception:
      source = type(self).__name__
    self.policy_bundle_hash = hashlib.sha256(source.encode("utf-8")).hexdigest()

  async def decide(
    self,
    *,
    payload: ApprovalRequestPayload,
    request: ApprovalRequest,
    run_context: RunContext,
  ) -> ApprovalDecision:
    _ = payload, run_context
    # An approval whose constraint is unknown (a row from before the column)
    # never executes (money, integrity): it is denied here and the agent
    # replans for a fresh approval.
    if request.approval_constraint == "legacy_unknown":
      return ApprovalDecision(
        outcome="auto_deny",
        reason="Approval constraint is unknown; replan and obtain a fresh approval",
        allow_persistent_grant=False,
        policy_id=self.policy_id,
        policy_version=self.policy_version,
      )
    # Fresh-owner requests retain their frozen human authority under every setting.
    if needs_human_approval(request.tool_class, request.approval_constraint):
      return ApprovalDecision(
        outcome="request_user_approval",
        reason=(
          "Exact promotion requires a fresh decision by its frozen owner"
          if request.approval_constraint == "fresh_human_owner"
          else f"{request.tool_class} tool requires explicit user approval"
        ),
        allow_persistent_grant=False,
        policy_id=self.policy_id,
        policy_version=self.policy_version,
      )
    return ApprovalDecision(
      outcome="auto_approve",
      reason="The operator approval policy requires no human decision for this class",
      allow_persistent_grant=False,
      policy_id=self.policy_id,
      policy_version=self.policy_version,
    )

  async def on_resolve(self, *, request: ApprovalRequest) -> None:
    return None

  async def revoke_persistent_grant(self, *, grant_id: str, reason: str) -> None:
    _ = reason
    if self._store is not None:
      await self._store.revoke_persistent_grant(grant_id)

  def role_authorized_for_class(self, *, decider_role: str | None, tool_class: str) -> bool:
    """Only the `owner` role may approve a live order.

    An order spends money in the owner's brokerage account; an `invite`
    collaborator shares the research desk, not that account, so its vote on an
    `irreversible` call is refused. Other classes retain their run's role authority.
    """

    if tool_class in MONEY_BOUNDARY_TOOL_CLASSES:
      return resolve_effective_role(decider_role) == "owner"
    return True
