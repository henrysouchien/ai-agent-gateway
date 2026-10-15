"""The init catalogue projects the authority's role catalogue and rates (design §7.5)."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import pytest
import yaml

from agent_gateway import server_chat_helpers
from agent_gateway.capability_execution import MaterializedCredential
from agent_gateway.server_chat_helpers import build_capability_choices
from agent_gateway.server_models import CapabilityRates
from agent_gateway.session import SessionStore
from model_authority.binding import CredentialHandle
from model_authority.capabilities import CAPABILITY_IDS, SESSION_DRIVER_CAPABILITY
from model_authority.loader import PACKAGED_AUTHORITY_DIR, ModelAuthority, load_model_authority


OPUS = "anthropic.claude-opus-5-5"
HAIKU = "anthropic.claude-haiku-5-5"
TENANT = "hank-test"


def _models(keys: list[str], effort: str) -> dict[str, Any]:
  return {"default": {"models": keys, "effort": effort}, "allowed_model_keys": keys}


def _authority(tmp_path: Path, *, unpriced: tuple[str, ...] = ()) -> ModelAuthority:
  # A candidate role map with session.driver on a role, synthesized the way
  # packages/model-authority/tests/test_model_roles.py does.
  document = yaml.safe_load((PACKAGED_AUTHORITY_DIR / "model-authority.yaml").read_text())
  sonnet = ["anthropic.claude-sonnet-5-5", "anthropic.claude-sonnet-5-5-sdk"]
  document["selection"]["tiers"] = {
    "deep": _models([OPUS], "high"),
    "default": {"default": {"alias": "deep"}, "allowed_model_keys": [OPUS, *sonnet]},
    "fast": _models([HAIKU, "anthropic.claude-haiku-5-5-sdk"], "none"),
    "standard": _models(sonnet, "none"),
    "plan": {"default": {"alias": "deep"}},
    "judge": {"default": {"alias": "fast"}},
    "vision": {"default": {"alias": "default"}},
    "quant": _models(["openai.gpt-6-1-sol"], "high"),
    "extract": _models(["openai.gpt-6-1-sol.chat"], "low"),
    "peer": _models(["openai.gpt-6-luna-sdk"], "none"),
    "ingest": _models(["anthropic.claude-opus-5-5-oauth"], "none"),
  }
  document["selection"].pop("presets", None)
  document["selection"]["capabilities"][SESSION_DRIVER_CAPABILITY]["default"] = {"kind": "tier", "tier": "default"}
  directory = tmp_path / "authority"
  shutil.copytree(PACKAGED_AUTHORITY_DIR, directory)
  (directory / "model-authority.yaml").write_text(yaml.safe_dump(document, sort_keys=False))
  rates_path = directory / "rates" / "anthropic.json"
  table = json.loads(rates_path.read_text())
  for model in unpriced:
    del table["providers"]["anthropic"]["models"][model]
  rates_path.write_text(json.dumps(table))
  return load_model_authority(directory)


def _session(authority: ModelAuthority):
  return SessionStore().create_session(
    api_key_hash="hash",
    user_id="alice",
    owner_user_id="101",
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


class _Provider:
  def __init__(self, name: str) -> None:
    self.name = name


def _builder(authority: ModelAuthority) -> Any:
  async def builder(*_args: Any, **_kwargs: Any) -> Any:
    raise AssertionError("the catalogue must not construct a runtime")

  def _materialize(handle: CredentialHandle) -> MaterializedCredential:
    raise AssertionError(f"the catalogue must not materialize a credential: {handle.provider}")

  adapters = {entry.adapter: entry.provider for entry in authority.registry.models.values()}
  setattr(builder, "_gateway_model_registry", authority.registry)
  setattr(builder, "_gateway_model_selection_policy", authority.selection_policy)
  setattr(builder, "_gateway_tenant_id", TENANT)
  setattr(builder, "_gateway_service_provider_handles", {})
  setattr(builder, "_gateway_service_auth_config_resolver", _materialize)
  setattr(builder, "_gateway_capability_adapter_resolver", lambda adapter: _Provider(adapters[adapter]))
  setattr(builder, "_gateway_channel_profile_allowlist", None)
  return builder


def _driver(authority: ModelAuthority, monkeypatch: pytest.MonkeyPatch):
  monkeypatch.setattr(server_chat_helpers, "RATE_TABLES", authority.rate_tables)
  return build_capability_choices(
    session=_session(authority), build_chat_runtime=_builder(authority),
  )[SESSION_DRIVER_CAPABILITY]


def test_driver_catalogue_lists_each_role_with_model_route_effort_and_rates(
  tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
  authority = _authority(tmp_path)

  driver = _driver(authority, monkeypatch)

  roles = {role.role: role for role in driver.roles}
  assert tuple(roles) == tuple(authority.selection_policy.tiers)
  deep = roles["deep"]
  assert (deep.model_key, deep.route, deep.effort, deep.unavailable) == (OPUS, "anthropic.public", "high", None)
  opus_rates = authority.rate_tables["anthropic"].lookup("anthropic", "claude-opus-5-5")
  assert isinstance(deep.rates, CapabilityRates)
  assert (deep.rates.input_cost_per_mtok, deep.rates.output_cost_per_mtok) == (
    opus_rates.input_cost_per_mtok, opus_rates.output_cost_per_mtok,
  )
  assert (roles["fast"].model_key, roles["fast"].effort) == (HAIKU, "none")
  # This session holds only an Anthropic credential: the OpenAI role names why,
  # with the code a real role request for it gets.
  assert roles["quant"].unavailable == "default_not_eligible"
  assert roles["quant"].effort is None
  assert "user-secret" not in driver.model_dump_json()


def test_driver_choices_carry_rates_without_widening(
  tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
  authority = _authority(tmp_path)

  driver = _driver(authority, monkeypatch)

  # The picker list is still the user-selectable, executable driver set.
  assert {choice.model_key for choice in driver.choices} <= set(
    authority.selection_policy.capabilities[SESSION_DRIVER_CAPABILITY].allowed_model_keys
  )
  assert all(
    authority.registry.require(choice.model_key).capabilities[SESSION_DRIVER_CAPABILITY] == "user_selectable"
    for choice in driver.choices
  )
  opus = next(choice for choice in driver.choices if choice.model_key == OPUS)
  assert isinstance(opus.rates, CapabilityRates)


def test_unpriced_model_reads_unknown_in_roles_and_choices(
  tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
  # Lookup falls back to the longest id prefix, so the Opus 5 row would price Opus 5.5.
  authority = _authority(tmp_path, unpriced=("claude-opus-5-5", "claude-opus-5"))

  driver = _driver(authority, monkeypatch)

  roles = {role.role: role for role in driver.roles}
  assert roles["deep"].model_key == OPUS
  assert roles["deep"].rates == "unknown"
  assert next(choice for choice in driver.choices if choice.model_key == OPUS).rates == "unknown"
  assert isinstance(roles["fast"].rates, CapabilityRates)
  wire = json.loads(driver.model_dump_json())
  assert next(role for role in wire["roles"] if role["role"] == "deep")["rates"] == "unknown"
