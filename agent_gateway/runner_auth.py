from __future__ import annotations

from typing import Any, Dict

from .auth import derive_auth_mechanism_and_entitlement

_SELECTION_AUTH_CONFIG_FIELDS = frozenset({
  "effort",
  "execution_transport",
  "model",
  "model_key",
  "thinking",
  "thinking_enabled_requested",
})

_CREDENTIAL_CLASS_FIELDS = ("auth_mechanism", "entitlement_kind")


def merge_refreshed_auth_config(
  config: Dict[str, Any],
  refreshed: Dict[str, Any],
) -> Dict[str, Any]:
  """Rotate credential material; execution selection and bound values stay.

  The execution bind owns model selection, provider and max_tokens. Auth mode
  belongs to the serving credential: refreshed material replaces it when
  supplied, otherwise the current mode remains. Provider pool fields
  (project_id, account_identity, auth_store_path, credential_source,
  refresh_token, token_expires_at, credential_affinity_key,
  credential_prepared) follow that same rule: preserve existing material,
  then replace fields supplied by the new seat. `auth_user_scope` is the
  exception: an existing trusted scope is pinned.
  Selection copies are dropped and bound values overwrite the resolver.

  `auth_mechanism` and `entitlement_kind` classify the serving material (the
  spend ledger's PAYG-vs-seat decision and placement eligibility read them). A
  rotation that changes the material without restamping them must not keep the
  previous credential's class: each one the refresh does not supply is
  re-derived from the merged material.
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
  # billing_mode and rate_table_version are the runner's bound values, and
  # auth_user_scope is the trusted owner of the pool: refreshed material
  # cannot change them.
  for key in ("billing_mode", "rate_table_version", "auth_user_scope"):
    if config.get(key):
      merged[key] = config[key]
  merged["api_key"] = str(merged.get("api_key", ""))
  merged["auth_token"] = str(merged.get("auth_token", ""))

  def _material(source: Dict[str, Any]) -> tuple[str, str, str]:
    return (
      str(source.get("api_key") or ""),
      str(source.get("auth_token") or ""),
      str(source.get("auth_mode", "api")).strip().lower(),
    )

  stale = [field for field in _CREDENTIAL_CLASS_FIELDS if field in config and field not in refreshed]
  if stale and _material(merged) != _material(config):
    live = {key: value for key, value in merged.items() if key not in _CREDENTIAL_CLASS_FIELDS}
    mechanism, entitlement = derive_auth_mechanism_and_entitlement(live)
    derived = {"auth_mechanism": mechanism, "entitlement_kind": entitlement}
    for field in stale:
      merged[field] = derived[field]
  return merged

