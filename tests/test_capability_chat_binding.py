from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError
from agent_gateway import AgentRunner

from model_authority.binding import CapabilityResolutionError, CredentialHandle
from agent_gateway.capability_execution import MaterializedCredential
from agent_gateway.event_log import EventLog
from model_authority.capabilities import CAPABILITY_IDS
from model_authority.current import INITIAL_MODEL_REGISTRY, INITIAL_MODEL_SELECTION_POLICY
from agent_gateway.providers import ModelInfo, ModelProvider
from agent_gateway.server import ChatRuntime, ChatTurnInputs
from agent_gateway.server_chat_helpers import (
  _dispatch_chat_turn,
  prepare_session_driver_turn,
)
from agent_gateway.server_models import ChatMessage, ChatRequest
from agent_gateway.session import SessionStore, SessionStream
from gateway_test_support.model_defaults import SESSION_DRIVER


class _ExactProvider(ModelProvider):
  def __init__(self, name: str) -> None:
    self.name = name

  def has_active_credential(self, config: dict[str, Any]) -> bool:
    return bool(config.get("api_key") or config.get("auth_token"))

  def get_model_info(self, model: str) -> ModelInfo:
    return ModelInfo(
      id=model,
      provider=self.name,
      max_output_tokens=64_000,
      supports_thinking=True,
    )


_PROVIDERS = {
  family: _ExactProvider(family)
  for family in {"anthropic", "codex", "openai", "xai"}
}


def _service_handle(provider: str) -> CredentialHandle:
  return CredentialHandle(
    handle_id=f"service:hank-test:{provider}",
    provider=provider,
    principal="service",
    tenant_id="hank-test",
    actor_id=None,
  )


def _session(*, auth_config: dict[str, Any] | None = None):
  return SessionStore().create_session(
    api_key_hash="hash",
    user_id="alice",
    owner_user_id="101",
    auth_config=(
      {
        "provider": "anthropic",
        "billing_mode": "byok",
        "auth_mode": "api",
        "api_key": "user-secret",
      }
      if auth_config is None
      else auth_config
    ),
    tenant_id="hank-test",
    credential_principal="user",
    allow_service_for_interactive=True,
    model_entitled_capabilities=CAPABILITY_IDS,
    model_entitled_keys=frozenset(INITIAL_MODEL_REGISTRY.models),
  )


def _inputs(
  *,
  model_key: str | None = None,
  effort: str | None = None,
  catalog_revision: str | None = None,
  context: dict[str, Any] | None = None,
) -> ChatTurnInputs:
  return ChatTurnInputs(
    messages=[ChatMessage(role="user", content="hello")],
    request_id="request-1",
    context=context or {},
    metadata={},
    model_key=model_key,
    effort=effort,
    catalog_revision=catalog_revision,
  )


def _configure_builder(builder: Any) -> None:
  service_handles = {
    provider: _service_handle(provider)
    for provider in {"codex", "openai", "xai"}
  }

  def _materialize(handle: CredentialHandle) -> MaterializedCredential:
    assert service_handles[handle.provider] is handle
    return MaterializedCredential(
      handle=handle,
      auth_config={
        "provider": handle.provider,
        "api_key": "service-secret",
        "auth_mode": "api",
        "billing_mode": "byok",
      },
    )

  setattr(builder, "_gateway_model_registry", INITIAL_MODEL_REGISTRY)
  setattr(
    builder,
    "_gateway_model_selection_policy",
    INITIAL_MODEL_SELECTION_POLICY,
  )
  setattr(builder, "_gateway_tenant_id", "hank-test")
  setattr(builder, "_gateway_service_provider_handles", service_handles)
  setattr(builder, "_gateway_service_auth_config_resolver", _materialize)
  setattr(
    builder,
    "_gateway_capability_adapter_resolver",
    lambda adapter: _PROVIDERS[
      INITIAL_MODEL_REGISTRY.models[
        next(
          key
          for key, entry in INITIAL_MODEL_REGISTRY.models.items()
          if entry.adapter == adapter
        )
      ].provider
    ],
  )
  setattr(builder, "_gateway_channel_profile_allowlist", None)


def test_prepare_keeps_auth_config_credential_only_and_binds_registry_identity() -> None:
  async def builder(*_args: Any, storage_root: Path | None = None, **_kwargs: Any) -> Any:
    raise AssertionError("preparation must not construct a runtime")

  _configure_builder(builder)
  prepared = prepare_session_driver_turn(
    _session(),
    _inputs(),
    build_chat_runtime=builder,
  )

  bind = prepared.request.capability_bind
  assert bind is not None
  assert bind.model_key == SESSION_DRIVER.model_key
  assert bind.upstream_model == SESSION_DRIVER.upstream_model
  assert bind.credential_principal == "user"
  assert prepared.request.bound_auth_config == {
    "provider": "anthropic",
    "billing_mode": "byok",
    "auth_mode": "api",
    "api_key": "user-secret",
  }
  assert "user-secret" not in repr(prepared.request)


def test_prepare_accepts_explicit_stable_key_with_service_authority() -> None:
  async def builder(*_args: Any, storage_root: Path | None = None, **_kwargs: Any) -> Any:
    raise AssertionError

  _configure_builder(builder)
  prepared = prepare_session_driver_turn(
    _session(),
    _inputs(model_key="openai.gpt-6-1-sol", effort="xhigh"),
    build_chat_runtime=builder,
  )
  bind = prepared.request.capability_bind
  assert bind is not None
  assert (
    bind.model_key,
    bind.provider,
    bind.upstream_model,
    bind.effort,
    bind.credential_principal,
  ) == (
    "openai.gpt-6-1-sol",
    "openai",
    "gpt-6.1-sol",
    "xhigh",
    "service",
  )


def test_prepare_rejects_provider_qualified_selector() -> None:
  async def builder(*_args: Any, storage_root: Path | None = None, **_kwargs: Any) -> Any:
    raise AssertionError

  _configure_builder(builder)
  with pytest.raises(CapabilityResolutionError) as refused:
    prepare_session_driver_turn(
      _session(),
      _inputs(model_key="openai:gpt-6.1-sol"),
      build_chat_runtime=builder,
    )
  assert refused.value.code == "capability_model_unavailable"


def test_prepare_with_stale_catalog_revision_and_ineligible_key_names_stale() -> None:
  async def builder(*_args: Any, storage_root: Path | None = None, **_kwargs: Any) -> Any:
    raise AssertionError

  _configure_builder(builder)
  with pytest.raises(CapabilityResolutionError) as refused:
    prepare_session_driver_turn(
      _session(),
      _inputs(
        model_key="anthropic.retired-model",
        catalog_revision="1999-01-01.0",
      ),
      build_chat_runtime=builder,
    )

  assert refused.value.code == "capability_catalog_stale"
  assert refused.value.catalog_revision == INITIAL_MODEL_REGISTRY.revision
  assert refused.value.eligible_model_keys
  receipt = refused.value.to_error()
  assert receipt["catalog_revision"] == INITIAL_MODEL_REGISTRY.revision
  assert "user-secret" not in str(receipt)


def test_prepare_with_stale_catalog_revision_and_eligible_key_still_binds() -> None:
  async def builder(*_args: Any, storage_root: Path | None = None, **_kwargs: Any) -> Any:
    raise AssertionError

  _configure_builder(builder)
  prepared = prepare_session_driver_turn(
    _session(),
    _inputs(
      model_key="anthropic.claude-sonnet-5-5",
      effort="xhigh",
      catalog_revision="1999-01-01.0",
    ),
    build_chat_runtime=builder,
  )

  bind = prepared.request.capability_bind
  assert bind is not None
  assert bind.model_key == "anthropic.claude-sonnet-5-5"
  assert bind.effort == "xhigh"


def test_effort_without_model_key_is_accepted() -> None:
  req = ChatRequest(
    messages=[ChatMessage(role="user", content="hello")],
    effort="high",
  )
  assert req.effort == "high"
  assert req.model_key is None


def test_context_authority_shaped_values_cannot_change_bind() -> None:
  async def builder(*_args: Any, storage_root: Path | None = None, **_kwargs: Any) -> Any:
    raise AssertionError

  _configure_builder(builder)
  prepared = prepare_session_driver_turn(
    _session(),
    _inputs(context={
      "model_key": "openai.gpt-6-1-sol",
      "provider": "openai",
      "capability_id": "plan.author",
      "credential_ref": "service:other",
      "run_mode": "batch",
    }),
    build_chat_runtime=builder,
  )
  bind = prepared.request.capability_bind
  assert bind is not None
  assert (
    bind.capability_id,
    bind.model_key,
    bind.provider,
    bind.credential_principal,
    bind.run_mode,
  ) == (
    "session.driver",
    SESSION_DRIVER.model_key,
    INITIAL_MODEL_REGISTRY.require(SESSION_DRIVER.model_key).provider,
    "user",
    "interactive",
  )


def test_prepare_projects_the_persisted_stage_route_without_changing_profile() -> None:
  async def builder(*_args: Any, storage_root: Path | None = None, **_kwargs: Any) -> Any:
    raise AssertionError

  _configure_builder(builder)
  session = _session()
  session.stage_skill_route = {
    "route_kind": "stage",
    "skill_name": "business-model-construction",
  }
  prepared = prepare_session_driver_turn(
    session,
    _inputs(context={
      "profile": "community",
      "skill": "caller-supplied-skill",
      "stage_skill_route": {
        "route_kind": "stage",
        "skill_name": "caller-supplied-skill",
      },
    }),
    build_chat_runtime=builder,
  )

  assert prepared.request.context["profile"] == "community"
  assert prepared.request.context["skill"] == "business-model-construction"
  assert prepared.request.context["stage_skill_route"] == {
    "route_kind": "stage",
    "skill_name": "business-model-construction",
  }


def test_prepare_never_falls_back_to_service_material_for_missing_user_secret() -> None:
  async def builder(*_args: Any, storage_root: Path | None = None, **_kwargs: Any) -> Any:
    raise AssertionError

  _configure_builder(builder)
  session = _session(auth_config={
    "provider": "anthropic",
    "billing_mode": "byok",
    "auth_mode": "api",
    "api_key": "",
  })
  with pytest.raises(CapabilityResolutionError) as refused:
    prepare_session_driver_turn(
      session,
      _inputs(),
      build_chat_runtime=builder,
    )
  assert refused.value.code == "credential_unavailable"


def test_prepared_request_can_be_bound_only_once() -> None:
  async def builder(*_args: Any, storage_root: Path | None = None, **_kwargs: Any) -> Any:
    raise AssertionError

  _configure_builder(builder)
  prepared = prepare_session_driver_turn(
    _session(),
    _inputs(),
    build_chat_runtime=builder,
  )
  capability_execution = prepared.request.capability_execution
  assert capability_execution is not None
  with pytest.raises(ValueError, match="already bound"):
    prepared.request._bind_session_driver(
      capability_execution=capability_execution,
    )


class _CompleteRunner(AgentRunner):
  def __init__(self, event_log: EventLog, execution: Any) -> None:
    self._event_log = event_log
    self._capability_execution = execution
    self._selected_content_bindings = ()
    self._selected_content_bindings_bound = False
    self.seen_messages: list[dict[str, Any]] = []

  async def run(
    self,
    messages,
    system_prompt=None,
    *,
    resume_initial_messages=None,
  ) -> None:
    _ = system_prompt, resume_initial_messages
    self.seen_messages = messages
    self._event_log.append({
      "type": "stream_complete",
      "terminal_disposition": "completed",
      "usage": {},
    })

@pytest.mark.asyncio
async def test_dispatch_emits_exact_complete_bind_before_runtime_work() -> None:
  seen_requests: list[Any] = []

  async def builder(
    session: Any,
    request: Any,
    channel: Any,
    auth_manager: Any,
    *,
    storage_root: Path | None = None,
  ):
    _ = session, channel, auth_manager
    seen_requests.append(request)
    return ChatRuntime(
      system_prompt="system",
      build_runner=lambda event_log, *_args: _CompleteRunner(
        event_log,
        request.capability_execution,
      ),
      capability_execution=request.capability_execution,
    )

  _configure_builder(builder)
  session = _session()
  inputs = _inputs()
  prepared = prepare_session_driver_turn(
    session,
    inputs,
    build_chat_runtime=builder,
  )
  events: list[dict[str, Any]] = []

  async def _capture(event: dict[str, Any]) -> None:
    events.append(event)

  result = await _dispatch_chat_turn(
    session,
    inputs,
    turn=SessionStream(event_log=EventLog(session_id=session.session_id), runner_task=None),
    on_event=_capture,
    build_chat_runtime=builder,
    transcript_dir=None,
    prepared_turn=prepared,
  )

  assert result.state == "completed"
  capability_bind = prepared.request.capability_bind
  assert capability_bind is not None
  assert seen_requests[0].capability_bind is capability_bind
  bound_events = [event for event in events if event.get("type") == "capability_bound"]
  assert len(bound_events) == 1
  assert bound_events[0] == {
    "type": "capability_bound",
    **capability_bind.to_json(),
  }
  assert "secret" not in str(bound_events[0])


@pytest.mark.asyncio
async def test_dispatch_attaches_turn_context_message_on_differ_only() -> None:
  # Excel workbook context belongs on the turn input (user message) on differ-only,
  # not in the cached static system prompt where every selection change rewrote
  # the entire system prompt and all conversation history.
  current_context: list[str] = ["## Context A"]
  last_runner: list[_CompleteRunner] = []

  async def builder(session: Any, request: Any, channel: Any, auth_manager: Any, *, storage_root: Path | None = None):
    _ = session, channel, auth_manager
    runner = _CompleteRunner(EventLog(session_id=session.session_id), request.capability_execution)
    last_runner.append(runner)
    return ChatRuntime(
      system_prompt="system",
      build_runner=lambda event_log, *_args: runner,
      capability_execution=request.capability_execution,
      turn_context_message=current_context[0],
    )

  _configure_builder(builder)
  session = _session()

  # Turn 1: Context A -> attached
  async def _noop(_: dict[str, Any]) -> None:
    pass

  # Turn 1: Context A -> attached
  inputs1 = _inputs(context={"workbook": "wb1"})
  inputs1.messages = [ChatMessage(role="user", content="msg 1")]
  prepared1 = prepare_session_driver_turn(session, inputs1, build_chat_runtime=builder)
  await _dispatch_chat_turn(
    session, inputs1,
    turn=SessionStream(event_log=EventLog(session_id=session.session_id), runner_task=None),
    on_event=_noop,
    build_chat_runtime=builder,
    transcript_dir=None,
    prepared_turn=prepared1,
  )
  assert last_runner[-1].seen_messages[-1]["content"] == "msg 1\n\n## Context A"
  assert session.last_turn_context_message == "## Context A"
  session.active_turn = None

  # Turn 2: Context A (unchanged selection) -> NOT attached
  inputs2 = _inputs(context={"workbook": "wb1"})
  inputs2.messages = [ChatMessage(role="user", content="msg 2")]
  prepared2 = prepare_session_driver_turn(session, inputs2, build_chat_runtime=builder)
  await _dispatch_chat_turn(
    session, inputs2,
    turn=SessionStream(event_log=EventLog(session_id=session.session_id), runner_task=None),
    on_event=_noop,
    build_chat_runtime=builder,
    transcript_dir=None,
    prepared_turn=prepared2,
  )
  assert last_runner[-1].seen_messages[-1]["content"] == "msg 2"
  session.active_turn = None

  # Turn 3: Context B (selection changed) -> attached
  current_context[0] = "## Context B"
  inputs3 = _inputs(context={"workbook": "wb1", "selection": "B5"})
  inputs3.messages = [ChatMessage(role="user", content="msg 3")]
  prepared3 = prepare_session_driver_turn(session, inputs3, build_chat_runtime=builder)
  await _dispatch_chat_turn(
    session, inputs3,
    turn=SessionStream(event_log=EventLog(session_id=session.session_id), runner_task=None),
    on_event=_noop,
    build_chat_runtime=builder,
    transcript_dir=None,
    prepared_turn=prepared3,
  )
  assert last_runner[-1].seen_messages[-1]["content"] == "msg 3\n\n## Context B"
  assert session.last_turn_context_message == "## Context B"


def test_session_creation_preserves_selected_anthropic_generation(tmp_path, monkeypatch):
  from agent_gateway.providers.anthropic_oauth import register_anthropic_static_credential

  monkeypatch.setenv("USER_DATA_DIR", str(tmp_path))
  def enroll(material):
    return register_anthropic_static_credential(
      material, source="request:anthropic:api_key", user_scope=101,
    )

  selected = enroll("first-key")
  assert selected is not None
  first_config = {
    "provider": "anthropic", "auth_mode": "api", "api_key": selected.token,
    "credential_source": selected.source_metadata(),
  }
  first = _session(auth_config=first_config)
  same = _session(auth_config=dict(first_config))
  replacement = enroll("replacement-key")
  assert replacement is not None
  changed = _session(auth_config={
    **first_config, "api_key": replacement.token,
    "credential_source": replacement.source_metadata(),
  })
  assert first.session_credential_handle is not None
  assert same.session_credential_handle is not None
  assert changed.session_credential_handle is not None
  assert first.session_credential_handle.handle_id == selected.generation_ref
  assert same.session_credential_handle == first.session_credential_handle
  assert changed.session_credential_handle.handle_id == replacement.generation_ref
  assert changed.session_credential_handle.handle_id != first.session_credential_handle.handle_id


def test_non_anthropic_sessions_keep_independent_handles():
  config = {
    "provider": "openai", "api_key": "same-key",
    "credential_source": {"generation_ref": "anthropic-generation-not-openai"},
  }
  first = _session(auth_config=config)
  second = _session(auth_config=config)
  assert first.session_credential_handle is not None
  assert second.session_credential_handle is not None
  assert first.session_credential_handle.handle_id != second.session_credential_handle.handle_id
