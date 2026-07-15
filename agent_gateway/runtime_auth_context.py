from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any


_AUTHENTICATED_RUNTIME_AUTH_CONFIG: ContextVar[dict[str, Any] | None] = ContextVar(
  "authenticated_runtime_auth_config",
  default=None,
)


def _copy_auth_config(auth_config: Any | None) -> dict[str, Any] | None:
  if auth_config is None:
    return None
  to_dict = getattr(auth_config, "to_dict", None)
  if callable(to_dict):
    auth_config = to_dict()
  if not isinstance(auth_config, Mapping):
    raise TypeError("authenticated runtime auth_config must be a mapping")
  return dict(auth_config)


@contextmanager
def bind_authenticated_runtime_auth_config(
  auth_config: Any | None,
) -> Iterator[None]:
  """Bind one authenticated control session's config to its async task tree."""

  token = _AUTHENTICATED_RUNTIME_AUTH_CONFIG.set(_copy_auth_config(auth_config))
  try:
    yield
  finally:
    _AUTHENTICATED_RUNTIME_AUTH_CONFIG.reset(token)


def current_authenticated_runtime_auth_config() -> dict[str, Any] | None:
  """Return a defensive copy of the task-scoped authenticated config."""

  return _copy_auth_config(_AUTHENTICATED_RUNTIME_AUTH_CONFIG.get())


__all__ = [
  "bind_authenticated_runtime_auth_config",
  "current_authenticated_runtime_auth_config",
]
