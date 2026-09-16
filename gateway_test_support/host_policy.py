from __future__ import annotations

from types import SimpleNamespace

import pytest

from agent_gateway.policy_imports import (
  configure_server_policy,
  load_server_policy_module,
)


@pytest.fixture
def owner_session_host_policy():
  """Authorize owner sessions without supplying application tool policy."""
  policy = SimpleNamespace(
    authority_denies_tool=lambda session, _tool_name, **_kwargs: (
      getattr(session, "role", None) != "owner"
    ),
    get_forbidden_tools_for_session=lambda _session: frozenset(),
    get_server_for_policy_tool=lambda _tool_name: None,
    get_tool_class=None,
  )
  previous = load_server_policy_module()
  configure_server_policy(policy)
  try:
    yield policy
  finally:
    configure_server_policy(previous)
