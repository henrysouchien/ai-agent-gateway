from __future__ import annotations

import inspect
from collections.abc import Callable
from typing import Any, Dict

MetricEmitter = Callable[[str, int], None]
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


async def call_credential_refresher(
  callback: Callable[[Any], Any] | None,
  failure: Any,
  *,
  emit_metric: MetricEmitter,
  log_session_id: str,
  logger: Any,
) -> Dict[str, Any] | None:
  if callback is None or not bool(getattr(failure, "retryable_with_new_credentials", False)):
    return None
  try:
    refreshed = callback(failure)
    if inspect.isawaitable(refreshed):
      refreshed = await refreshed
  except Exception as exc:
    logger.warning(
      "[%s] credential refresh failed after provider %s failure (non-fatal): %s",
      log_session_id,
      getattr(failure, "kind", ""),
      exc,
    )
    emit_metric("gateway.credential_refresh_failed", 1)
    return None
  if not isinstance(refreshed, dict) or not refreshed:
    emit_metric("gateway.credential_refresh_unavailable", 1)
    return None
  return dict(refreshed)
