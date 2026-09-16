import json
from dataclasses import replace
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
PKG_DIR = Path(__file__).resolve().parents[1]
if str(PKG_DIR) not in sys.path:
  sys.path.insert(0, str(PKG_DIR))

from agent_gateway.rates import (
  ModelRates,
  UnknownModelError,
  load_rate_table,
  resolve_configured_rates_file,
)
from agent_gateway.model_registry import INITIAL_MODEL_REGISTRY
from agent_gateway.providers import AnthropicProvider, CodexProvider, OpenAIProvider, XAIProvider
from agent_gateway.runner_budget import CostAccumulator, admit_provider_request_budget


def _write_rate_table(tmp_path: Path, payload: dict) -> Path:
  path = tmp_path / "rates.json"
  path.write_text(json.dumps(payload), encoding="utf-8")
  return path


def _base_payload(*, version: str = "2026-04-08", models: dict[str, dict] | None = None) -> dict:
  return {
    "version": version,
    "source": "https://example.test/pricing",
    "providers": {
      "anthropic": {
        "models": models
        or {
          "claude-sonnet-4-6": {
            "display_name": "Claude Sonnet 4.6",
            "input_cost_per_mtok": 3.0,
            "output_cost_per_mtok": 15.0,
            "cache_read_cost_per_mtok": 0.3,
            "cache_write_cost_per_mtok": 3.75,
            "max_tokens": 16384,
            "context_window": 200000,
          }
        }
      }
    },
  }


@pytest.fixture(autouse=True)
def _clear_rates_env(monkeypatch: pytest.MonkeyPatch) -> None:
  monkeypatch.delenv("AGENT_GATEWAY_RATES_FILE", raising=False)


def test_load_rate_table_none_loads_bundled_default() -> None:
  table = load_rate_table(None)

  assert table.version
  assert table.source == "https://platform.claude.com/docs/en/about-claude/models/overview"
  assert table.providers["anthropic"]


def test_load_rate_table_explicit_path_loads_that_file(tmp_path: Path) -> None:
  path = _write_rate_table(tmp_path, _base_payload(version="explicit-version"))

  table = load_rate_table(path)

  assert table.version == "explicit-version"


def test_load_rate_table_env_var_override(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
  path = _write_rate_table(tmp_path, _base_payload(version="env-version"))
  monkeypatch.setenv("AGENT_GATEWAY_RATES_FILE", str(path))

  table = load_rate_table(None)

  assert table.version == "env-version"


def test_load_rate_table_kwarg_wins_over_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
  env_path = _write_rate_table(tmp_path, _base_payload(version="env-version"))
  kwarg_path = tmp_path / "kwarg-rates.json"
  kwarg_path.write_text(json.dumps(_base_payload(version="kwarg-version")), encoding="utf-8")
  monkeypatch.setenv("AGENT_GATEWAY_RATES_FILE", str(env_path))

  table = load_rate_table(kwarg_path)

  assert table.version == "kwarg-version"


def test_explicit_rate_path_precedes_invalid_relative_env(
  monkeypatch: pytest.MonkeyPatch,
  tmp_path: Path,
) -> None:
  path = _write_rate_table(tmp_path, _base_payload(version="explicit-version"))
  monkeypatch.setenv("AGENT_GATEWAY_RATES_FILE", "relative/rates.json")

  assert load_rate_table(path).version == "explicit-version"


def test_configured_rates_file_rejects_relative_path() -> None:
  with pytest.raises(
    ValueError,
    match="AGENT_GATEWAY_RATES_FILE must configure an absolute path",
  ):
    resolve_configured_rates_file({
      "AGENT_GATEWAY_RATES_FILE": "relative/rates.json",
    })


def test_configured_rates_file_expands_user_identically(
  monkeypatch: pytest.MonkeyPatch,
  tmp_path: Path,
) -> None:
  monkeypatch.setenv("HOME", str(tmp_path))
  configured = resolve_configured_rates_file({
    "AGENT_GATEWAY_RATES_FILE": "~/rates.json",
  })

  assert configured == tmp_path / "rates.json"


def test_configured_rates_file_unset_or_blank_returns_none() -> None:
  assert resolve_configured_rates_file({}) is None
  assert resolve_configured_rates_file({"AGENT_GATEWAY_RATES_FILE": "  "}) is None


def test_lookup_exact_match_returns_model_rates() -> None:
  table = load_rate_table(None)

  rates = table.lookup("anthropic", "claude-sonnet-4-6")

  assert isinstance(rates, ModelRates)
  assert rates.display_name == "Claude Sonnet 4.6"
  assert rates.input_cost_per_mtok == 3.0
  assert rates.output_cost_per_mtok == 15.0


def test_lookup_fable_returns_bundled_model_rates() -> None:
  table = load_rate_table(None)

  rates = table.lookup("anthropic", "claude-fable-5")

  assert rates.display_name == "Claude Fable 5"
  assert rates.input_cost_per_mtok == 10.0
  assert rates.output_cost_per_mtok == 50.0
  assert rates.cache_read_cost_per_mtok == 1.0
  assert rates.cache_write_cost_per_mtok == 12.5
  assert rates.max_tokens == 128_000
  assert rates.context_window == 1_000_000


def test_lookup_opus48_returns_bundled_model_rates() -> None:
  table = load_rate_table(None)

  rates = table.lookup("anthropic", "claude-opus-4-8")

  assert rates.display_name == "Claude Opus 4.8"
  assert rates.input_cost_per_mtok == 5.0
  assert rates.output_cost_per_mtok == 25.0
  assert rates.cache_read_cost_per_mtok == 0.5
  assert rates.cache_write_cost_per_mtok == 6.25
  assert rates.max_tokens == 128_000
  assert rates.context_window == 1_000_000


def test_lookup_opus5_returns_bundled_model_rates() -> None:
  table = load_rate_table(None)

  rates = table.lookup("anthropic", "claude-opus-5")

  assert rates.display_name == "Claude Opus 5"
  assert rates.input_cost_per_mtok == 5.0
  assert rates.output_cost_per_mtok == 25.0
  assert rates.cache_read_cost_per_mtok == 0.5
  assert rates.cache_write_cost_per_mtok == 6.25
  assert rates.max_tokens == 128_000
  assert rates.context_window == 1_000_000


def test_lookup_tag_match_returns_shorter_tag_entry(tmp_path: Path) -> None:
  path = _write_rate_table(
    tmp_path,
    _base_payload(
      models={
        "claude-sonnet": {
          "display_name": "Tag Match",
          "input_cost_per_mtok": 1.0,
          "output_cost_per_mtok": 2.0,
          "cache_read_cost_per_mtok": 0.1,
          "cache_write_cost_per_mtok": 0.2,
          "max_tokens": 1000,
          "context_window": 2000,
        }
      }
    ),
  )
  table = load_rate_table(path)

  rates = table.lookup("anthropic", "claude-sonnet-4-6-20250514")

  assert rates.display_name == "Tag Match"


def test_lookup_prefix_match_returns_canonical_entry(tmp_path: Path) -> None:
  path = _write_rate_table(tmp_path, _base_payload())
  table = load_rate_table(path)

  rates = table.lookup("anthropic", "claude-sonnet-4-6-20250514")

  assert rates.display_name == "Claude Sonnet 4.6"


def test_lookup_unknown_model_raises(tmp_path: Path) -> None:
  path = _write_rate_table(tmp_path, _base_payload(version="test-version"))
  table = load_rate_table(path)

  with pytest.raises(UnknownModelError):
    table.lookup("anthropic", "claude-unknown")


def test_load_rate_table_malformed_json_fails_fast(tmp_path: Path) -> None:
  path = tmp_path / "bad-rates.json"
  path.write_text("{", encoding="utf-8")

  with pytest.raises(ValueError, match=r"malformed JSON"):
    load_rate_table(path)


def test_load_rate_table_missing_version_field_raises_clear_error(tmp_path: Path) -> None:
  path = _write_rate_table(tmp_path, {"source": "https://example.test/pricing", "providers": {}})

  with pytest.raises(ValueError, match=r"missing required top-level field 'version'"):
    load_rate_table(path)


def test_lookup_prefers_exact_then_longest_model_prefix(tmp_path: Path) -> None:
  path = _write_rate_table(
    tmp_path,
    _base_payload(
      models={
        "claude-sonnet-4-6-20250514": {
          "display_name": "Exact Match",
          "input_cost_per_mtok": 9.0,
          "output_cost_per_mtok": 9.0,
          "cache_read_cost_per_mtok": 0.9,
          "cache_write_cost_per_mtok": 0.9,
          "max_tokens": 9000,
          "context_window": 9000,
        },
        "claude-sonnet": {
          "display_name": "Tag Match",
          "input_cost_per_mtok": 1.0,
          "output_cost_per_mtok": 1.0,
          "cache_read_cost_per_mtok": 0.1,
          "cache_write_cost_per_mtok": 0.1,
          "max_tokens": 1000,
          "context_window": 1000,
        },
        "claude-sonnet-4-6": {
          "display_name": "Prefix Match",
          "input_cost_per_mtok": 2.0,
          "output_cost_per_mtok": 2.0,
          "cache_read_cost_per_mtok": 0.2,
          "cache_write_cost_per_mtok": 0.2,
          "max_tokens": 2000,
          "context_window": 2000,
        },
      }
    ),
  )
  table = load_rate_table(path)

  assert table.lookup("anthropic", "claude-sonnet-4-6-20250514").input_cost_per_mtok == 9.0
  assert table.lookup("anthropic", "claude-sonnet-4-6-latest").input_cost_per_mtok == 2.0
  assert table.lookup("anthropic", "anthropic/claude-sonnet-4-6-latest").input_cost_per_mtok == 2.0


@pytest.mark.parametrize(
  ("provider_name", "model"),
  [
    ("codex", "gpt-6-astra"),
    ("openai", "gpt-6-astra"),
    ("xai", "grok-4.6"),
  ],
)
def test_configured_rates_price_registry_only_models(
  provider_name: str, model: str, monkeypatch, tmp_path: Path,
) -> None:
  provider_type = {"codex": CodexProvider, "openai": OpenAIProvider, "xai": XAIProvider}[provider_name]
  model_rates = {
    "display_name": model,
    "input_cost_per_mtok": 2.0,
    "output_cost_per_mtok": 3.0,
    "cache_read_cost_per_mtok": 4.0,
    "cache_write_cost_per_mtok": 5.0,
    "max_tokens": 3200,
    "context_window": 211000,
  }
  path = _write_rate_table(tmp_path, {
    "version": "deployment-override",
    "source": "https://example.test/pricing",
    "providers": {provider_name: {"models": {model: model_rates}}},
  })
  monkeypatch.setenv("AGENT_GATEWAY_RATES_FILE", str(path))
  provider = provider_type()

  info = provider.get_model_info(model)
  assert info.context_window == 211000
  assert info.max_output_tokens == 3200
  estimate = provider.estimate_cost(
    model, 1000, 2000, cache_read_tokens=3000, cache_creation_tokens=4000,
  )
  assert estimate.total == pytest.approx(0.04)


@pytest.mark.parametrize("provider_type", [OpenAIProvider, CodexProvider])
@pytest.mark.parametrize("injected", [False, True])
def test_anthropic_override_preserves_other_provider_budget(
  provider_type, injected: bool, monkeypatch, tmp_path: Path,
) -> None:
  path = _write_rate_table(tmp_path, _base_payload())
  monkeypatch.setenv("AGENT_GATEWAY_RATES_FILE", str(path))
  provider = provider_type(rate_table=load_rate_table(path)) if injected else provider_type()

  assert provider.estimate_cost("gpt-5.6", 100_000, 1_000).total == pytest.approx(0.53)
  admission = admit_provider_request_budget(
    CostAccumulator(0.50), provider=provider, model="gpt-5.6",
    estimated_input_tokens=100_000, requested_max_output_tokens=1_000,
  )
  assert admission.denied_state is not None


@pytest.mark.parametrize(
  ("provider_type", "model", "input_tokens", "expected"),
  [
    (XAIProvider, "grok-4.6", 250_000, 1.012),
    (OpenAIProvider, "gpt-6-astra", 300_000, 6.075),
    (CodexProvider, "gpt-6-astra", 300_000, 6.075),
  ],
)
def test_published_long_context_prices(provider_type, model, input_tokens, expected) -> None:
  assert provider_type().estimate_cost(model, input_tokens, 1_000).total == pytest.approx(expected)
  admission = admit_provider_request_budget(
    CostAccumulator(0.60 if provider_type is XAIProvider else 4.0),
    provider=provider_type(), model=model, estimated_input_tokens=input_tokens,
    requested_max_output_tokens=1_000,
  )
  assert admission.denied_state is not None


@pytest.mark.parametrize(
  ("provider_type", "model", "input_tokens", "expected"),
  [
    (XAIProvider, "grok-4.6", 199_999, 0.405998),
    (XAIProvider, "grok-4.6", 200_000, 0.812),
    (OpenAIProvider, "gpt-6-astra", 272_000, 2.77),
    (OpenAIProvider, "gpt-6-astra", 272_001, 5.51502),
  ],
)
def test_long_context_threshold_inclusivity(provider_type, model, input_tokens, expected) -> None:
  assert provider_type().estimate_cost(model, input_tokens, 1_000).total == pytest.approx(expected)


@pytest.mark.parametrize(
  ("provider_type", "model", "uncached", "cached", "written", "expected"),
  [
    (XAIProvider, "grok-4.6", 100_000, 100_000, 0, 0.512),
    (OpenAIProvider, "gpt-6-astra", 172_000, 100_000, 0, 1.87),
    (OpenAIProvider, "gpt-6-astra", 172_001, 100_000, 0, 3.71502),
    (CodexProvider, "gpt-6-astra", 0, 0, 300_000, 7.575),
  ],
)
def test_long_context_tier_counts_all_prompt_tokens(
  provider_type, model, uncached, cached, written, expected,
) -> None:
  estimate = provider_type().estimate_cost(
    model, uncached, 1_000, cache_read_tokens=cached, cache_creation_tokens=written,
  )
  assert estimate.total == pytest.approx(expected)


def test_partial_override_preserves_other_models_and_providers(monkeypatch, tmp_path: Path) -> None:
  payload = _base_payload()
  payload["providers"]["anthropic"]["models"]["claude-sonnet-4-6"]["input_cost_per_mtok"] = 7
  path = _write_rate_table(tmp_path, payload)
  monkeypatch.setenv("AGENT_GATEWAY_RATES_FILE", str(path))

  assert AnthropicProvider().estimate_cost("claude-sonnet-4-6", 100_000, 1_000).total == pytest.approx(0.715)
  assert AnthropicProvider().estimate_cost("claude-fable-5", 100_000, 1_000).total == pytest.approx(1.05)
  assert XAIProvider().estimate_cost("grok-4.6", 100_000, 1_000).total == pytest.approx(0.206)


def test_override_prefix_precedes_exact_bundled_prices_and_environment(monkeypatch, tmp_path: Path) -> None:
  payload = _base_payload()
  row = payload["providers"]["anthropic"]["models"].pop("claude-sonnet-4-6")
  row["input_cost_per_mtok"] = 7
  payload["providers"]["anthropic"]["models"]["claude-sonnet"] = row
  override = load_rate_table(_write_rate_table(tmp_path, payload))
  monkeypatch.setenv("AGENT_GATEWAY_RATES_FILE", "invalid-relative-path.json")

  provider = AnthropicProvider(rate_table=override)
  assert provider.estimate_cost("claude-sonnet-4-6", 100_000, 1_000).total == pytest.approx(0.715)


@pytest.mark.parametrize("provider_type", [CodexProvider, OpenAIProvider, XAIProvider])
def test_unknown_rate_model_keeps_warn_and_zero(provider_type, monkeypatch, caplog) -> None:
  import agent_gateway.providers.base as provider_base

  template = next(entry for entry in INITIAL_MODEL_REGISTRY.models.values() if entry.provider == provider_type.name)
  model = "unpriced-future-model"
  entry = replace(
    template, key=f"{provider_type.name}.unpriced", upstream_model=model,
    reported_identities=frozenset({model}),
  )
  monkeypatch.setattr(
    provider_base, "INITIAL_MODEL_REGISTRY",
    replace(INITIAL_MODEL_REGISTRY, models={entry.key: entry}),
  )

  assert provider_type().estimate_cost(model, 1000, 2000, cache_read_tokens=3000).total == 0
  assert any(record.levelname == "WARNING" and model in record.getMessage() for record in caplog.records)
