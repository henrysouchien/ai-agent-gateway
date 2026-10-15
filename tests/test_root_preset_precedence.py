"""The root turn captures one whole winning preset map: run > user > verified firm > policy.

The run level is the session's frozen `/chat/init` ``model_preset`` (model-roles W4d).
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any
from uuid import UUID

import pytest
import yaml

from agent_gateway.capability_execution import MaterializedCredential
from agent_gateway.model_preferences import ModelPreferenceStore
from agent_gateway.providers import ModelInfo, ModelProvider
from agent_gateway.server import ChatTurnInputs
from agent_gateway.server_chat_helpers import (
  bind_init_capability_selections,
  build_capability_choices,
  prepare_session_driver_turn,
)
from agent_gateway.server_models import ChatMessage, PreparedChatTurn
from agent_gateway.session import SessionStore, bind_session_capability_selections
from model_authority.binding import CapabilityResolutionError, CredentialHandle
from model_authority.capabilities import CAPABILITY_IDS
from model_authority.loader import PACKAGED_AUTHORITY_DIR, ModelAuthority, load_model_authority
from model_authority.selection import SelectionSnapshot, TierModelsDefault


OPUS = "anthropic.claude-opus-5-5"
SONNET = "anthropic.claude-sonnet-5-5"
HAIKU = "anthropic.claude-haiku-5-5"
HAIKU_SDK = "anthropic.claude-haiku-5-5-sdk"
CODEX_SOL = "codex.gpt-6-1-sol"
FIRM = UUID("11111111-1111-4111-8111-111111111111")
TENANT = "hank-test"
ACTOR = "101"  # the session owner: the preference store's actor key


def _authority(tmp_path: Path) -> ModelAuthority:
  # A candidate artifact publishing presets, synthesized the way
  # packages/model-authority/tests/test_model_roles.py does.
  document = yaml.safe_load((PACKAGED_AUTHORITY_DIR / "model-authority.yaml").read_text())
  document["selection"]["presets"] = {
    "user_economy": {"default": {"models": [SONNET], "effort": "low"}},
    "firm_standard": {
      "default": {"models": [OPUS], "effort": "medium"},
      "fast": {"models": [HAIKU, HAIKU_SDK], "effort": "low"},
    },
    "codex_driver": {"default": {"models": [CODEX_SOL], "effort": "high"}},
    "codex_with_fast": {
      "default": {"models": [CODEX_SOL], "effort": "high"},
      "fast": {"models": [HAIKU, HAIKU_SDK], "effort": "low"},
    },
    "run_eval": {"default": {"models": [SONNET], "effort": "medium"}},
  }
  directory = tmp_path / "authority"
  shutil.copytree(PACKAGED_AUTHORITY_DIR, directory)
  (directory / "model-authority.yaml").write_text(yaml.safe_dump(document, sort_keys=False))
  return load_model_authority(directory)


def _session(authority: ModelAuthority):
  return SessionStore().create_session(
    api_key_hash="hash",
    user_id="alice",
    owner_user_id=ACTOR,
    auth_config={
      "provider": "anthropic",
      "billing_mode": "byok",
      "auth_mode": "api",
      "api_key": "user-secret",
    },
    tenant_id=TENANT,
    credential_principal="user",
    allow_service_for_interactive=False,
    model_entitled_capabilities=CAPABILITY_IDS,
    model_entitled_keys=frozenset(authority.registry.models),
  )


def _inputs(*, model_key: str | None = None, effort: str | None = None) -> ChatTurnInputs:
  return ChatTurnInputs(
    messages=[ChatMessage(role="user", content="hello")],
    request_id="request-1",
    context={},
    metadata={},
    model_key=model_key,
    effort=effort,
  )


class _Provider(ModelProvider):
  def __init__(self, name: str) -> None:
    self.name = name

  def has_active_credential(self, config: dict[str, Any]) -> bool:
    return bool(config.get("api_key") or config.get("auth_token"))

  def get_model_info(self, model: str) -> ModelInfo:
    return ModelInfo(id=model, provider=self.name, max_output_tokens=64_000, supports_thinking=True)


def _builder(authority: ModelAuthority, store: ModelPreferenceStore) -> Any:
  async def builder(*_args: Any, **_kwargs: Any) -> Any:
    raise AssertionError("preparation must not construct a runtime")

  def _materialize(handle: CredentialHandle) -> MaterializedCredential:
    raise AssertionError(f"no service credential is configured: {handle.provider}")

  adapters = {entry.adapter: entry.provider for entry in authority.registry.models.values()}
  setattr(builder, "_gateway_model_registry", authority.registry)
  setattr(builder, "_gateway_model_selection_policy", authority.selection_policy)
  setattr(builder, "_gateway_tenant_id", TENANT)
  setattr(builder, "_gateway_service_provider_handles", {})
  setattr(builder, "_gateway_service_auth_config_resolver", _materialize)
  setattr(builder, "_gateway_capability_adapter_resolver", lambda adapter: _Provider(adapters[adapter]))
  setattr(builder, "_gateway_channel_profile_allowlist", None)
  setattr(builder, "_gateway_model_preference_store", store)
  return builder


@pytest.fixture
def authority(tmp_path: Path) -> ModelAuthority:
  return _authority(tmp_path)


@pytest.fixture
def store(tmp_path: Path, authority: ModelAuthority) -> ModelPreferenceStore:
  return ModelPreferenceStore(
    tmp_path / "preferences.sqlite3",
    known_preset_names=frozenset(authority.selection_policy.presets),
  )


def _select_user(store: ModelPreferenceStore, authority: ModelAuthority, name: str) -> None:
  store.put_user_preset(
    tenant_id=TENANT, actor_id=ACTOR, name=name,
    policy_revision=authority.selection_policy.revision,
  )


def _select_firm(store: ModelPreferenceStore, authority: ModelAuthority, name: str) -> None:
  store.put_firm_preset(
    commercial_account_id=FIRM, name=name,
    policy_revision=authority.selection_policy.revision,
  )


def _prepare(authority: ModelAuthority, store: ModelPreferenceStore, **kwargs: Any) -> PreparedChatTurn:
  inputs = kwargs.pop("inputs", None) or _inputs()
  return prepare_session_driver_turn(
    _session(authority),
    inputs,
    build_chat_runtime=_builder(authority, store),
    **kwargs,
  )


def _driver(prepared: PreparedChatTurn) -> tuple[str, str, str]:
  bind = prepared.request.capability_bind
  assert bind is not None
  return bind.model_key, bind.effort, bind.selection_source


def _snapshot(prepared: PreparedChatTurn) -> SelectionSnapshot:
  snapshot = prepared.selection_snapshot
  assert snapshot is not None
  return snapshot


def test_policy_applies_with_no_selection(authority: ModelAuthority, store: ModelPreferenceStore) -> None:
  prepared = _prepare(authority, store, commercial_account_id=FIRM)

  assert _driver(prepared) == (OPUS, "high", "capability_default")
  snapshot = _snapshot(prepared)
  assert snapshot.overridden == frozenset()
  assert snapshot.source is None
  assert prepared.request.selection_snapshot is snapshot


def test_verified_firm_preset_beats_policy(authority: ModelAuthority, store: ModelPreferenceStore) -> None:
  _select_firm(store, authority, "firm_standard")

  prepared = _prepare(authority, store, commercial_account_id=FIRM)

  assert _driver(prepared) == (OPUS, "medium", "internal_policy")
  snapshot = _snapshot(prepared)
  assert snapshot.source == "internal_policy"
  assert snapshot.assignments["fast"] == TierModelsDefault((HAIKU, HAIKU_SDK), "low")


def test_firm_layer_needs_a_verified_commercial_account(authority: ModelAuthority, store: ModelPreferenceStore) -> None:
  _select_firm(store, authority, "firm_standard")

  prepared = _prepare(authority, store)

  assert _driver(prepared) == (OPUS, "high", "capability_default")
  assert _snapshot(prepared).overridden == frozenset()


def test_user_preset_beats_firm_and_omitted_roles_fill_from_policy(
  authority: ModelAuthority, store: ModelPreferenceStore,
) -> None:
  _select_firm(store, authority, "firm_standard")
  _select_user(store, authority, "user_economy")

  prepared = _prepare(authority, store, commercial_account_id=FIRM)

  assert _driver(prepared) == (SONNET, "low", "saved_preference")
  snapshot = _snapshot(prepared)
  assert snapshot.source == "saved_preference"
  # The user map omits `fast`: it comes from policy, never the firm's override.
  policy_fast = authority.selection_policy.tiers["fast"].default
  assert isinstance(policy_fast, TierModelsDefault)
  assert snapshot.assignments["fast"] == TierModelsDefault(policy_fast.models, policy_fast.effort)
  assert snapshot.assignments["fast"] == TierModelsDefault((HAIKU, HAIKU_SDK), "none")
  assert "fast" not in snapshot.overridden
  assert snapshot.assignments["deep"] == TierModelsDefault((OPUS,), "high")


def test_explicit_effort_stays_exact_under_a_preset(authority: ModelAuthority, store: ModelPreferenceStore) -> None:
  _select_user(store, authority, "user_economy")

  prepared = _prepare(
    authority, store, inputs=_inputs(model_key=SONNET, effort="xhigh"),
  )

  assert _driver(prepared) == (SONNET, "xhigh", "explicit_user")
  # The captured map still governs later children of this root turn.
  snapshot = _snapshot(prepared)
  assert snapshot.source == "saved_preference"
  assert snapshot.assignments["default"] == TierModelsDefault((SONNET,), "low")


def test_unpublished_saved_preset_is_discarded_with_notice_and_base_policy_applies(
  authority: ModelAuthority, store: ModelPreferenceStore,
) -> None:
  _select_firm(store, authority, "firm_standard")
  _select_user(store, authority, "withdrawn_preset")
  builder = _builder(authority, store)

  prepared = prepare_session_driver_turn(
    _session(authority), _inputs(), build_chat_runtime=builder,
    commercial_account_id=FIRM,
  )

  # Base policy, not the firm preset (R5).
  assert _driver(prepared) == (OPUS, "high", "capability_default")
  assert _snapshot(prepared).overridden == frozenset()
  driver = build_capability_choices(session=_session(authority), build_chat_runtime=builder)["session.driver"]
  notice = next(notice for notice in driver.notices if notice.code == "saved_preset_not_applied")
  assert notice.reason == "preset_unpublished"
  assert "'withdrawn_preset'" in notice.message
  assert "it remains saved until replaced or cleared" in notice.message
  assert store.get_user_preset(tenant_id=TENANT, actor_id=ACTOR) is not None


def test_saved_preset_without_a_driver_credential_is_discarded_with_notice(
  authority: ModelAuthority, store: ModelPreferenceStore,
) -> None:
  _select_firm(store, authority, "firm_standard")
  _select_user(store, authority, "codex_driver")
  builder = _builder(authority, store)

  prepared = prepare_session_driver_turn(
    _session(authority), _inputs(), build_chat_runtime=builder,
    commercial_account_id=FIRM,
  )

  assert _driver(prepared) == (OPUS, "high", "capability_default")
  assert _snapshot(prepared).overridden == frozenset()
  assert prepared.request.selection_snapshot is prepared.selection_snapshot
  driver = build_capability_choices(session=_session(authority), build_chat_runtime=builder)["session.driver"]
  notice = next(notice for notice in driver.notices if notice.code == "saved_preset_not_applied")
  assert notice.reason is not None and notice.reason != "preset_unpublished"


def test_no_notice_when_the_saved_preset_applies(authority: ModelAuthority, store: ModelPreferenceStore) -> None:
  _select_user(store, authority, "user_economy")
  builder = _builder(authority, store)

  driver = build_capability_choices(session=_session(authority), build_chat_runtime=builder)["session.driver"]

  assert all(notice.code != "saved_preset_not_applied" for notice in driver.notices)


def test_role_catalogue_lists_what_the_saved_preset_binds(authority: ModelAuthority, store: ModelPreferenceStore) -> None:
  _select_user(store, authority, "user_economy")

  driver = build_capability_choices(
    session=_session(authority), build_chat_runtime=_builder(authority, store),
  )["session.driver"]
  prepared = _prepare(authority, store)

  default = next(role for role in driver.roles if role.role == "default")
  assert (default.model_key, default.effort) == (SONNET, "low")
  assert _driver(prepared)[:2] == (default.model_key, default.effort)


@pytest.mark.parametrize("preset", ["user_economy", "codex_driver"])
def test_init_selected_is_what_the_turn_binds_under_a_saved_preset(
  authority: ModelAuthority, store: ModelPreferenceStore, preset: str,
) -> None:
  # `user_economy` applies; `codex_driver` has no driver credential and is discarded.
  _select_user(store, authority, preset)

  driver = build_capability_choices(
    session=_session(authority), build_chat_runtime=_builder(authority, store),
  )["session.driver"]
  prepared = _prepare(authority, store)

  selected = driver.selected
  assert selected is not None
  assert (selected.model_key, selected.effort, selected.reason) == _driver(prepared)


def test_discarded_saved_preset_leaves_init_roles_on_base_policy(
  authority: ModelAuthority, store: ModelPreferenceStore,
) -> None:
  # The session holds only an Anthropic credential, so the preset's Codex
  # driver cannot bind: the turn discards the whole preset, and init's roles,
  # `selected` and notice must say the same.
  _select_user(store, authority, "codex_with_fast")

  driver = build_capability_choices(
    session=_session(authority), build_chat_runtime=_builder(authority, store),
  )["session.driver"]
  prepared = _prepare(authority, store)

  assert _driver(prepared) == (OPUS, "high", "capability_default")
  snapshot = _snapshot(prepared)
  assert snapshot.overridden == frozenset()
  roles = {role.role: role for role in driver.roles}
  assert (roles["default"].model_key, roles["default"].effort, roles["default"].unavailable) == (OPUS, "high", None)
  fast = snapshot.assignments["fast"]
  assert isinstance(fast, TierModelsDefault)
  assert fast.effort == "none"
  assert roles["fast"].effort == "none"
  selected = driver.selected
  assert selected is not None
  assert (selected.model_key, selected.effort, selected.reason) == _driver(prepared)
  assert [notice.code for notice in driver.notices].count("saved_preset_not_applied") == 1



def test_unpublished_firm_preset_fails_explicitly(authority: ModelAuthority, store: ModelPreferenceStore) -> None:
  _select_firm(store, authority, "withdrawn_preset")

  with pytest.raises(CapabilityResolutionError) as refused:
    _prepare(authority, store, commercial_account_id=FIRM)

  assert refused.value.code == "capability_selection_invalid"
  assert "withdrawn_preset" in str(refused.value)


def _run_session(authority: ModelAuthority, store: ModelPreferenceStore, name: str):
  session = _session(authority)
  bind_init_capability_selections(
    session=session,
    selections={},
    build_chat_runtime=_builder(authority, store),
    model_preset=name,
  )
  return session


def test_run_preset_beats_saved_user_and_firm_presets_and_omitted_roles_fill_from_policy(
  authority: ModelAuthority, store: ModelPreferenceStore,
) -> None:
  _select_firm(store, authority, "firm_standard")
  _select_user(store, authority, "user_economy")
  session = _run_session(authority, store, "run_eval")

  prepared = prepare_session_driver_turn(
    session, _inputs(), build_chat_runtime=_builder(authority, store), commercial_account_id=FIRM,
  )

  assert _driver(prepared) == (SONNET, "medium", "internal_policy")
  snapshot = _snapshot(prepared)
  assert snapshot.source == "internal_policy"
  # `fast` comes from policy, never the user's or the firm's preset.
  assert "fast" not in snapshot.overridden
  assert snapshot.assignments["fast"] == TierModelsDefault((HAIKU, HAIKU_SDK), "none")


def test_init_selected_is_what_the_turn_binds_under_a_run_preset(
  authority: ModelAuthority, store: ModelPreferenceStore,
) -> None:
  # The saved `codex_driver` would be discarded; the run level never consults it.
  _select_user(store, authority, "codex_driver")
  session = _run_session(authority, store, "run_eval")
  builder = _builder(authority, store)

  driver = build_capability_choices(session=session, build_chat_runtime=builder)["session.driver"]
  prepared = prepare_session_driver_turn(session, _inputs(), build_chat_runtime=builder)

  selected = driver.selected
  assert selected is not None
  assert (selected.model_key, selected.effort, selected.reason) == _driver(prepared)
  assert _driver(prepared) == (SONNET, "medium", "internal_policy")
  assert all(notice.code != "saved_preset_not_applied" for notice in driver.notices)


def test_unpublished_run_preset_is_refused_at_init(authority: ModelAuthority, store: ModelPreferenceStore) -> None:
  session = _session(authority)

  with pytest.raises(CapabilityResolutionError) as refused:
    bind_init_capability_selections(
      session=session,
      selections={},
      build_chat_runtime=_builder(authority, store),
      model_preset="withdrawn_preset",
    )

  assert refused.value.code == "capability_selection_invalid"
  assert session.model_preset is None


def test_unpublished_run_preset_fails_the_turn_without_falling_back(
  authority: ModelAuthority, store: ModelPreferenceStore,
) -> None:
  # A name frozen at init that the serving policy no longer publishes.
  _select_user(store, authority, "user_economy")
  session = _session(authority)
  bind_session_capability_selections(session, run_overrides={}, model_preset="withdrawn_preset")

  with pytest.raises(CapabilityResolutionError) as refused:
    prepare_session_driver_turn(session, _inputs(), build_chat_runtime=_builder(authority, store))

  assert refused.value.code == "capability_selection_invalid"
