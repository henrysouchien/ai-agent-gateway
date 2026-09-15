from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Mapping


_DEFAULT_RATE_TABLE_PATH = Path(__file__).with_name("rates") / "anthropic.json"
_RATES_FILE_ENV_VAR = "AGENT_GATEWAY_RATES_FILE"


class UnknownModelError(ValueError):
  """Raised when a requested model is missing from the configured rate table."""


@dataclass(frozen=True)
class ContextRateTier:
  """Per-million-token prices for a whole request at or above its prompt threshold."""

  min_input_tokens: int
  input_cost_per_mtok: float
  output_cost_per_mtok: float
  cache_read_cost_per_mtok: float
  cache_write_cost_per_mtok: float


@dataclass(frozen=True)
class ModelRates:
  display_name: str
  input_cost_per_mtok: float
  output_cost_per_mtok: float
  cache_read_cost_per_mtok: float
  cache_write_cost_per_mtok: float
  max_tokens: int | None
  context_window: int | None
  tiers: tuple[ContextRateTier, ...] = ()


@dataclass(frozen=True)
class RateTable:
  version: str
  source: str
  providers: dict[str, dict[str, ModelRates]]
  _path: Path = field(repr=False, compare=False)
  _bundled: RateTable | None = field(default=None, repr=False, compare=False)

  def lookup(self, provider: str, model: str) -> ModelRates:
    provider_name = str(provider or "").strip().lower()
    model_id = str(model or "").strip()
    provider_models = self.providers.get(provider_name, {})
    if model_id in provider_models:
      return provider_models[model_id]

    candidates = (model_id, model_id.rsplit("/", 1)[-1])
    match: ModelRates | None = None
    match_length = -1
    for key, rates in provider_models.items():
      if len(key) > match_length and any(
        candidate == key or candidate.startswith(f"{key}-") for candidate in candidates
      ):
        match = rates
        match_length = len(key)
    if match is not None:
      return match
    if self._bundled is not None:
      return self._bundled.lookup(provider_name, model_id)

    raise UnknownModelError(
      f"Model '{model_id}' not in rate table v{self.version}. Update {self._path} or pass --rates-file with a newer version."
    )


def _expect_mapping(value: Any, *, path: Path, label: str) -> dict[str, Any]:
  if not isinstance(value, dict):
    raise ValueError(f"Rate table file {path} has invalid {label}: expected an object.")
  return value


def _expect_required_field(mapping: dict[str, Any], key: str, *, path: Path, label: str) -> Any:
  if key not in mapping:
    raise ValueError(f"Rate table file {path} is missing required {label} field '{key}'.")
  return mapping[key]


def _parse_optional_int(value: Any, *, path: Path, label: str) -> int | None:
  if value is None:
    return None
  if isinstance(value, bool) or not isinstance(value, int):
    raise ValueError(f"Rate table file {path} has invalid {label}: expected an integer or null.")
  return value


def _parse_required_float(value: Any, *, path: Path, label: str) -> float:
  if isinstance(value, bool) or not isinstance(value, (int, float)):
    raise ValueError(f"Rate table file {path} has invalid {label}: expected a number.")
  return float(value)


def _parse_required_str(value: Any, *, path: Path, label: str) -> str:
  if not isinstance(value, str) or not value.strip():
    raise ValueError(f"Rate table file {path} has invalid {label}: expected a non-empty string.")
  return value


def _parse_tiers(path: Path, label: str, raw_tiers: Any) -> tuple[ContextRateTier, ...]:
  if not isinstance(raw_tiers, list):
    raise ValueError(f"Rate table file {path} has invalid {label}: expected an array.")
  tiers = []
  for index, raw_tier in enumerate(raw_tiers):
    tier_label = f"{label}[{index}]"
    raw = _expect_mapping(raw_tier, path=path, label=tier_label)
    threshold = _parse_optional_int(
      _expect_required_field(raw, "min_input_tokens", path=path, label=tier_label),
      path=path, label=f"{tier_label}.min_input_tokens",
    )
    if threshold is None or threshold < 0:
      raise ValueError(f"Rate table file {path} has invalid {tier_label}.min_input_tokens: expected a non-negative integer.")
    prices = {
      key: _parse_required_float(
        _expect_required_field(raw, key, path=path, label=tier_label),
        path=path, label=f"{tier_label}.{key}",
      )
      for key in (
        "input_cost_per_mtok", "output_cost_per_mtok",
        "cache_read_cost_per_mtok", "cache_write_cost_per_mtok",
      )
    }
    tiers.append(ContextRateTier(min_input_tokens=threshold, **prices))
  return tuple(sorted(tiers, key=lambda tier: tier.min_input_tokens, reverse=True))


def _parse_model_rates(path: Path, provider: str, model: str, raw_model: Any) -> ModelRates:
  raw = _expect_mapping(raw_model, path=path, label=f"providers.{provider}.models.{model}")
  return ModelRates(
    display_name=_parse_required_str(
      _expect_required_field(raw, "display_name", path=path, label=f"providers.{provider}.models.{model}"),
      path=path,
      label=f"providers.{provider}.models.{model}.display_name",
    ),
    input_cost_per_mtok=_parse_required_float(
      _expect_required_field(raw, "input_cost_per_mtok", path=path, label=f"providers.{provider}.models.{model}"),
      path=path,
      label=f"providers.{provider}.models.{model}.input_cost_per_mtok",
    ),
    output_cost_per_mtok=_parse_required_float(
      _expect_required_field(raw, "output_cost_per_mtok", path=path, label=f"providers.{provider}.models.{model}"),
      path=path,
      label=f"providers.{provider}.models.{model}.output_cost_per_mtok",
    ),
    cache_read_cost_per_mtok=_parse_required_float(
      _expect_required_field(
        raw,
        "cache_read_cost_per_mtok",
        path=path,
        label=f"providers.{provider}.models.{model}",
      ),
      path=path,
      label=f"providers.{provider}.models.{model}.cache_read_cost_per_mtok",
    ),
    cache_write_cost_per_mtok=_parse_required_float(
      _expect_required_field(
        raw,
        "cache_write_cost_per_mtok",
        path=path,
        label=f"providers.{provider}.models.{model}",
      ),
      path=path,
      label=f"providers.{provider}.models.{model}.cache_write_cost_per_mtok",
    ),
    max_tokens=_parse_optional_int(raw.get("max_tokens"), path=path, label=f"providers.{provider}.models.{model}.max_tokens"),
    context_window=_parse_optional_int(
      raw.get("context_window"),
      path=path,
      label=f"providers.{provider}.models.{model}.context_window",
    ),
    tiers=_parse_tiers(path, f"providers.{provider}.models.{model}.tiers", raw.get("tiers", [])),
  )


def resolve_configured_rates_file(
  env: Mapping[str, str] | None = None,
) -> Path | None:
  """Resolve the configured rates file without cwd-dependent behavior."""

  environ = os.environ if env is None else env
  raw_path = str(environ.get(_RATES_FILE_ENV_VAR) or "").strip()
  if not raw_path:
    return None
  path = Path(raw_path).expanduser()
  if not path.is_absolute():
    raise ValueError(
      f"{_RATES_FILE_ENV_VAR} must configure an absolute path: {path}"
    )
  return path


def load_rate_table(path: Path | None = None) -> RateTable:
  selected_path = path
  if selected_path is None:
    selected_path = resolve_configured_rates_file() or _DEFAULT_RATE_TABLE_PATH

  try:
    raw_payload = json.loads(selected_path.read_text(encoding="utf-8"))
  except json.JSONDecodeError as exc:
    raise ValueError(f"Rate table file {selected_path} is malformed JSON: {exc.msg}.") from exc
  except OSError as exc:
    raise ValueError(f"Failed to read rate table file {selected_path}: {exc}.") from exc

  raw_table = _expect_mapping(raw_payload, path=selected_path, label="root payload")
  version = _parse_required_str(
    _expect_required_field(raw_table, "version", path=selected_path, label="top-level"),
    path=selected_path,
    label="top-level version",
  )
  source = _parse_required_str(
    _expect_required_field(raw_table, "source", path=selected_path, label="top-level"),
    path=selected_path,
    label="top-level source",
  )
  raw_providers = _expect_mapping(
    _expect_required_field(raw_table, "providers", path=selected_path, label="top-level"),
    path=selected_path,
    label="top-level providers",
  )

  providers: dict[str, dict[str, ModelRates]] = {}
  for provider_name, raw_provider in raw_providers.items():
    provider_block = _expect_mapping(raw_provider, path=selected_path, label=f"providers.{provider_name}")
    raw_models = _expect_mapping(
      _expect_required_field(provider_block, "models", path=selected_path, label=f"providers.{provider_name}"),
      path=selected_path,
      label=f"providers.{provider_name}.models",
    )
    providers[str(provider_name).strip().lower()] = {
      str(model_name): _parse_model_rates(selected_path, str(provider_name), str(model_name), raw_model)
      for model_name, raw_model in raw_models.items()
    }

  return RateTable(version=version, source=source, providers=providers, _path=selected_path)


def load_provider_rate_table(provider: str, rate_table: RateTable | None = None) -> RateTable:
  """Apply only this provider's override rows, retaining bundled prices for absent rows.

  Explicit tables take precedence over the environment. Override lookup runs
  first, so configured model-prefix prices also override exact bundled rows.
  """
  provider_name = str(provider).strip().lower()
  bundled = load_rate_table(_DEFAULT_RATE_TABLE_PATH.with_name(f"{provider_name}.json"))
  if rate_table is None:
    configured_path = resolve_configured_rates_file()
    if configured_path is not None:
      rate_table = load_rate_table(configured_path)
  if rate_table is None or not rate_table.providers.get(provider_name):
    return bundled
  return replace(rate_table, _bundled=bundled)


__all__ = [
  "ContextRateTier",
  "ModelRates",
  "RateTable",
  "UnknownModelError",
  "load_provider_rate_table",
  "load_rate_table",
  "resolve_configured_rates_file",
]
