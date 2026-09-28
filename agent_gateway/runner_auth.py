from __future__ import annotations

from typing import Any, Dict

_SELECTION_AUTH_CONFIG_FIELDS = frozenset({
  "effort",
  "execution_transport",
  "model",
  "model_key",
  "thinking",
  "thinking_enabled_requested",
})


def merge_refreshed_auth_config(
  config: Dict[str, Any],
  refreshed: Dict[str, Any],
) -> Dict[str, Any]:
  """Rotate credential material without reopening execution selection."""

  duplicated = sorted(
    _SELECTION_AUTH_CONFIG_FIELDS & (set(config) | set(refreshed))
  )
  if duplicated:
    raise ValueError(
      "credential refresh material must not contain model selection: "
      + ", ".join(duplicated)
    )
  provider = str(config.get("provider") or "").strip().lower()

  immutable_values = {
    "auth_mode": str(config.get("auth_mode", "api")).strip().lower(),
    "max_tokens": int(config.get("max_tokens", 16000)),
  }
  if provider:
    immutable_values["provider"] = provider
  for key, expected in immutable_values.items():
    if key not in refreshed:
      continue
    candidate = refreshed[key]
    if key in {"provider", "auth_mode"}:
      candidate = str(candidate or "").strip().lower()
    elif key == "max_tokens":
      candidate = int(candidate)
    if candidate != expected:
      raise ValueError(
        f"credential refresh cannot change bound {key}"
      )
  merged = dict(config)
  merged.update(refreshed)
  merged.update(immutable_values)
  for key in ("billing_mode", "rate_table_version"):
    if config.get(key):
      merged[key] = config[key]
  merged["api_key"] = str(merged.get("api_key", ""))
  merged["auth_token"] = str(merged.get("auth_token", ""))
  return merged

