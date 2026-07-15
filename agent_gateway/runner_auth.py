from __future__ import annotations

import inspect
from collections.abc import Callable
from typing import Any, Dict
from .thinking import canonical_effort_config


MetricEmitter = Callable[[str, int], None]


def merge_refreshed_auth_config(config: Dict[str, Any], refreshed: Dict[str, Any]) -> Dict[str, Any]:
  # Preserve request-scoped model/runtime controls. The refresh hook rotates
  # credentials for the already-selected provider; model changes belong to a
  # new session/init handshake.
  canonical_config = canonical_effort_config(config)
  preserved = {
    "model": canonical_config.get("model"),
    "max_tokens": canonical_config.get("max_tokens"),
    "effort": canonical_config.get("effort"),
  }
  merged = dict(canonical_config)
  merged.update(refreshed)
  for key, value in preserved.items():
    if value is not None:
      merged[key] = value
  # Refreshed credentials may come from an older resolver that still emits
  # the deprecated boolean alias. The persisted canonical effort wins and
  # must not be reinterpreted as a same-layer dual-key pair.
  merged.pop("thinking", None)
  for key in ("billing_mode", "rate_table_version"):
    if config.get(key):
      merged[key] = config[key]
  merged["auth_mode"] = str(merged.get("auth_mode", "api")).strip().lower()
  merged["api_key"] = str(merged.get("api_key", ""))
  merged["auth_token"] = str(merged.get("auth_token", ""))
  merged["model"] = str(merged.get("model") or "")
  merged["max_tokens"] = int(merged.get("max_tokens", 16000))
  return canonical_effort_config(merged)


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
