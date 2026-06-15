import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
PKG_DIR = ROOT / "packages" / "agent-gateway"
if str(PKG_DIR) not in sys.path:
  sys.path.insert(0, str(PKG_DIR))

from agent_gateway.rates import ModelRates, UnknownModelError, load_rate_table


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
  assert table.source == "https://docs.anthropic.com/pricing"
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
  assert rates.max_tokens == 32_000
  assert rates.context_window == 1_000_000


def test_lookup_opus48_returns_bundled_model_rates() -> None:
  table = load_rate_table(None)

  rates = table.lookup("anthropic", "claude-opus-4-8")

  assert rates.display_name == "Claude Opus 4.8"
  assert rates.input_cost_per_mtok == 5.0
  assert rates.output_cost_per_mtok == 25.0
  assert rates.cache_read_cost_per_mtok == 0.5
  assert rates.cache_write_cost_per_mtok == 6.25
  assert rates.max_tokens == 32_000
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


def test_lookup_unknown_model_raises_actionable_error(tmp_path: Path) -> None:
  path = _write_rate_table(tmp_path, _base_payload(version="test-version"))
  table = load_rate_table(path)

  with pytest.raises(
    UnknownModelError,
    match=(
      rf"Model 'claude-unknown' not in rate table vtest-version\. "
      rf"Update {path} or pass --rates-file with a newer version\."
    ),
  ):
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


def test_lookup_match_order_exact_then_tag_then_prefix(tmp_path: Path) -> None:
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

  assert table.lookup("anthropic", "claude-sonnet-4-6-20250514").display_name == "Exact Match"
  assert table.lookup("anthropic", "claude-sonnet-4-6-latest").display_name == "Tag Match"
