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
  """Rotate credential material; execution selection and bound values stay.

  The execution bind owns model selection, provider and max_tokens. Auth mode
  belongs to the serving credential: refreshed material replaces it when
  supplied, otherwise the current mode remains. Provider pool fields
  (project_id, account_identity, auth_store_path, auth_user_scope,
  credential_source, refresh_token, token_expires_at) follow that same rule:
  preserve existing material, then replace fields supplied by the new seat.
  Selection copies are dropped and bound values overwrite the resolver.
  """

  provider = str(config.get("provider") or "").strip().lower()

  bound_values: Dict[str, Any] = {
    "max_tokens": int(config.get("max_tokens", 16000)),
  }
  if provider:
    bound_values["provider"] = provider
  merged = {
    key: value
    for key, value in config.items()
    if key not in _SELECTION_AUTH_CONFIG_FIELDS
  }
  merged.update(
    (key, value)
    for key, value in refreshed.items()
    if key not in _SELECTION_AUTH_CONFIG_FIELDS
  )
  merged["auth_mode"] = str(merged.get("auth_mode", "api")).strip().lower()
  merged.update(bound_values)
  for key in ("billing_mode", "rate_table_version"):
    if config.get(key):
      merged[key] = config[key]
  merged["api_key"] = str(merged.get("api_key", ""))
  merged["auth_token"] = str(merged.get("auth_token", ""))
  return merged

