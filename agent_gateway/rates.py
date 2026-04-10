from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


_DEFAULT_RATE_TABLE_PATH = Path(__file__).with_name("rates") / "anthropic.json"
_RATES_FILE_ENV_VAR = "AGENT_GATEWAY_RATES_FILE"


class UnknownModelError(ValueError):
  """Raised when a requested model is missing from the configured rate table."""


@dataclass(frozen=True)
class ModelRates:
  display_name: str
  input_cost_per_mtok: float
  output_cost_per_mtok: float
  cache_read_cost_per_mtok: float
  cache_write_cost_per_mtok: float
  max_tokens: int | None
  context_window: int | None


@dataclass(frozen=True)
class RateTable:
  version: str
  source: str
  providers: dict[str, dict[str, ModelRates]]
  _path: Path = field(repr=False, compare=False)

  def lookup(self, provider: str, model: str) -> ModelRates:
    provider_name = str(provider or "").strip().lower()
    model_id = str(model or "").strip()
    provider_models = self.providers.get(provider_name, {})
    if model_id in provider_models:
      return provider_models[model_id]

    for key, rates in provider_models.items():
      if key in model_id:
        return rates

    candidates = [model_id]
    if "/" in model_id:
      candidates.append(model_id.rsplit("/", 1)[-1])
    for key, rates in provider_models.items():
      if any(candidate == key or candidate.startswith(f"{key}-") for candidate in candidates):
        return rates

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
  )


def load_rate_table(path: Path | None = None) -> RateTable:
  selected_path = path
  if selected_path is None:
    env_path = os.environ.get(_RATES_FILE_ENV_VAR, "").strip()
    selected_path = Path(env_path) if env_path else _DEFAULT_RATE_TABLE_PATH

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


__all__ = ["ModelRates", "RateTable", "UnknownModelError", "load_rate_table"]
