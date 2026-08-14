from __future__ import annotations

import copy
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from .capability_binding import CapabilityBind


MarkerPosition = tuple[int, int]


def _freeze(value: Any) -> Any:
  if isinstance(value, Mapping):
    return MappingProxyType({
      str(key): _freeze(item)
      for key, item in value.items()
    })
  if isinstance(value, (list, tuple)):
    return tuple(_freeze(item) for item in value)
  if isinstance(value, set):
    return frozenset(_freeze(item) for item in value)
  return copy.deepcopy(value)


def _thaw(value: Any) -> Any:
  if isinstance(value, Mapping):
    return {
      str(key): _thaw(item)
      for key, item in value.items()
    }
  if isinstance(value, tuple):
    return [_thaw(item) for item in value]
  if isinstance(value, frozenset):
    return {_thaw(item) for item in value}
  return copy.deepcopy(value)


def _required_text(value: object, *, field_name: str) -> str:
  text = str(value or "").strip()
  if not text:
    raise ValueError(f"{field_name} must be non-empty")
  return text


@dataclass(frozen=True, slots=True)
class ForkRequestHandoff:
  """Immutable request-identity snapshot used to seed one fork child."""

  _messages: tuple[Any, ...]
  rendered_system_blocks: tuple[tuple[str, bool], ...]
  _wire_tools: tuple[Any, ...]
  max_tokens: int
  _auth_config: Mapping[str, Any] = field(repr=False)
  capability_bind: CapabilityBind
  tenant_id: str
  billing_mode: str
  message_marker_position: MarkerPosition
  boundary_kind: str

  def __post_init__(self) -> None:
    object.__setattr__(self, "_messages", _freeze(self._messages))
    object.__setattr__(self, "_wire_tools", _freeze(self._wire_tools))
    object.__setattr__(self, "_auth_config", _freeze(self._auth_config))
    object.__setattr__(
      self,
      "rendered_system_blocks",
      tuple(
        (
          _required_text(block[0], field_name="system block text"),
          bool(block[1]),
        )
        for block in self.rendered_system_blocks
      ),
    )
    if self.boundary_kind not in {"mid_turn", "post_turn"}:
      raise ValueError("fork handoff boundary_kind must be mid_turn or post_turn")
    if (
      isinstance(self.max_tokens, bool)
      or not isinstance(self.max_tokens, int)
      or self.max_tokens <= 0
    ):
      raise ValueError("fork handoff max_tokens must be a positive integer")
    if (
      not isinstance(self.message_marker_position, tuple)
      or len(self.message_marker_position) != 2
      or any(
        isinstance(index, bool) or not isinstance(index, int) or index < 0
        for index in self.message_marker_position
      )
    ):
      raise ValueError(
        "fork handoff message_marker_position must be two non-negative integers"
      )
    for field_name in (
      "tenant_id",
      "billing_mode",
    ):
      object.__setattr__(
        self,
        field_name,
        _required_text(getattr(self, field_name), field_name=field_name),
      )
    if not isinstance(self.capability_bind, CapabilityBind):
      raise TypeError("fork handoff capability_bind must be a CapabilityBind")
    if self.capability_bind.capability_id != "session.driver":
      raise ValueError("fork handoff must snapshot the session.driver bind")
    auth_config = self.auth_config
    duplicated_selection = {
      "effort",
      "model",
      "model_key",
      "thinking",
      "thinking_enabled_requested",
    } & set(auth_config)
    if duplicated_selection:
      raise ValueError(
        "fork handoff auth material contains model selection"
      )
    configured_provider = str(
      auth_config.get("provider") or ""
    ).strip().lower()
    if configured_provider and configured_provider != self.capability_bind.provider:
      raise ValueError("fork handoff auth provider does not match its bind")

  @property
  def messages(self) -> list[dict[str, Any]]:
    return _thaw(self._messages)

  @property
  def wire_tools(self) -> list[dict[str, Any]]:
    return _thaw(self._wire_tools)

  @property
  def auth_config(self) -> dict[str, Any]:
    return _thaw(self._auth_config)


def _selected_credential_identity(
  runner: Any,
) -> tuple[str, str] | None:
  bind = runner._capability_execution.bind
  session = (
    getattr(runner, "_gateway_session", None)
    or getattr(getattr(runner, "_dispatcher", None), "_session", None)
  )
  handle = getattr(session, "session_credential_handle", None)
  if handle is not None:
    handle_id = str(
      getattr(handle, "handle_id", "") or ""
    ).strip()
    handle_provider = str(
      getattr(handle, "provider", "") or ""
    ).strip().lower()
    handle_principal = str(
      getattr(handle, "principal", "") or ""
    ).strip().lower()
    if (
      handle_id != bind.credential_ref
      or handle_provider != str(bind.provider).strip().lower()
      or handle_principal
      != str(bind.credential_principal).strip().lower()
    ):
      return None
    tenant_id = str(
      getattr(handle, "tenant_id", "") or ""
    ).strip()
    return (
      (handle_id, tenant_id)
      if handle_id and tenant_id
      else None
    )

  child_tenant_id = str(
    getattr(runner, "_tenant_id", "") or ""
  ).strip()
  if child_tenant_id:
    return bind.credential_ref, child_tenant_id

  auth_config = runner._capability_execution.auth_config
  auth_handle_id = str(
    auth_config.get("credential_handle_id", "") or ""
  ).strip()
  auth_tenant_id = str(
    auth_config.get("tenant_id", "") or ""
  ).strip()
  if auth_handle_id == bind.credential_ref and auth_tenant_id:
    return auth_handle_id, auth_tenant_id
  return None


def credential_identity_or_none(
  runner: Any,
) -> tuple[str, str] | None:
  return _selected_credential_identity(runner)


def _credential_identity(runner: Any) -> tuple[str, str]:
  identity = _selected_credential_identity(runner)
  if identity is None:
    raise ValueError(
      "fork handoff requires the parent's bound credential handle and tenant"
    )
  return identity


def _snapshot_request_identity(
  runner: Any,
) -> tuple[
  tuple[tuple[str, bool], ...],
  tuple[Any, ...],
  MarkerPosition,
]:
  raw_system = getattr(runner, "_last_request_system_blocks", None)
  raw_tools = getattr(runner, "_last_request_wire_tools", None)
  raw_marker = getattr(runner, "_last_request_message_marker_position", None)
  if not isinstance(raw_system, (list, tuple)):
    raise ValueError("fork handoff requires rendered request system blocks")
  if not isinstance(raw_tools, (list, tuple)):
    raise ValueError("fork handoff requires the request wire tool list")
  if (
    not isinstance(raw_marker, tuple)
    or len(raw_marker) != 2
  ):
    raise ValueError("fork handoff requires a parent message cache marker")
  system_blocks = tuple(
    (
      _required_text(block[0], field_name="system block text"),
      bool(block[1]),
    )
    for block in raw_system
  )
  return system_blocks, _freeze(raw_tools), raw_marker


def _build_handoff(
  runner: Any,
  messages: Sequence[Mapping[str, Any]],
  *,
  boundary_kind: str,
) -> ForkRequestHandoff:
  system_blocks, wire_tools, marker = _snapshot_request_identity(runner)
  execution = runner._capability_execution
  execution.validate()
  credential_handle_id, tenant_id = _credential_identity(runner)
  config = dict(execution.auth_config)
  max_tokens = int(
    getattr(runner, "_last_request_max_tokens", None)
    or getattr(runner, "_max_tokens_override", None)
    or config.get("max_tokens")
    or 0
  )
  return ForkRequestHandoff(
    _messages=_freeze(messages),
    rendered_system_blocks=system_blocks,
    _wire_tools=wire_tools,
    max_tokens=max_tokens,
    _auth_config=_freeze(config),
    capability_bind=execution.bind,
    tenant_id=tenant_id,
    billing_mode=str(getattr(runner, "_billing_mode", "") or ""),
    message_marker_position=marker,
    boundary_kind=boundary_kind,
  )


def build_mid_turn_handoff(
  runner: Any,
  messages: Sequence[Mapping[str, Any]],
) -> ForkRequestHandoff:
  """Snapshot history ending with the in-flight assistant tool-use message."""

  return _build_handoff(runner, messages, boundary_kind="mid_turn")


def build_post_turn_handoff(
  runner: Any,
  messages: Sequence[Mapping[str, Any]],
  final_assistant_message: Mapping[str, Any],
) -> ForkRequestHandoff:
  """Snapshot a proven terminal turn, explicitly retaining its final assistant."""

  completed = copy.deepcopy(list(messages))
  final_copy = copy.deepcopy(dict(final_assistant_message))
  latest_assistant = next(
    (
      message
      for message in reversed(completed)
      if message.get("role") == "assistant"
    ),
    None,
  )
  if latest_assistant != final_copy:
    completed.append(final_copy)
  return _build_handoff(runner, completed, boundary_kind="post_turn")


__all__ = [
  "ForkRequestHandoff",
  "MarkerPosition",
  "build_mid_turn_handoff",
  "build_post_turn_handoff",
  "credential_identity_or_none",
]
