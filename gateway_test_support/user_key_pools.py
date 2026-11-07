from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any, Iterator, Mapping
from unittest.mock import patch



class StaticUserKeyPools:
  def __init__(self, auth_config: Mapping[str, Any], *, identity_role: str | None = None) -> None:
    self._auth_config = dict(auth_config)
    self._identity_role = identity_role
    self.lease_calls: list[dict[str, Any]] = []

  def enrolled(self, **_kwargs: Any) -> bool:
    return True

  def lease(self, **kwargs: Any) -> Any:
    self.lease_calls.append(dict(kwargs))
    return SimpleNamespace(
      auth_config=lambda: dict(self._auth_config),
      identity_role=self._identity_role,
    )


@contextmanager
def static_user_key_pools(
  auth_config: Mapping[str, Any], *, identity_role: str | None = None,
) -> Iterator[StaticUserKeyPools]:
  pools = StaticUserKeyPools(auth_config, identity_role=identity_role)
  with patch("credentials_resolver.installed_user_key_pools", lambda _provider=None: pools):
    yield pools
