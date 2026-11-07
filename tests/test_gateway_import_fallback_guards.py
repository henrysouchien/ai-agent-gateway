from __future__ import annotations

import logging
from types import SimpleNamespace
from typing import Any

import pytest

from agent_gateway.control_plane import profiles as profiles_module




def test_control_profiles_are_empty_without_an_application_owner() -> None:
  assert profiles_module._list_profile_metadata(
    profile_names_provider=None,
    profile_loader=None,
  ) == []


def test_control_profiles_use_canonical_profile_loader(
  caplog: pytest.LogCaptureFixture,
) -> None:
  calls: list[str] = []

  def load_profile(name: str) -> Any:
    calls.append(name)
    if name == "broken":
      raise RuntimeError("broken profile")
    return SimpleNamespace(
      name=name,
      channel_context=f"{name}-channel",
      supports_autonomous_execution=True,
    )

  with caplog.at_level(logging.WARNING, logger="agent_gateway.control_plane.profiles"):
    entries = profiles_module._list_profile_metadata(
      profile_names_provider=lambda: ("research", "broken", "advisor"),
      profile_loader=load_profile,
    )

  assert calls == ["research", "broken", "advisor"]
  assert [(entry.name, entry.channel_context) for entry in entries] == [
    ("advisor", "advisor-channel"),
    ("research", "research-channel"),
  ]
  assert "profile broken failed to load" in caplog.text


def test_control_profiles_require_names_and_loader_together() -> None:
  with pytest.raises(ValueError, match="must be configured together"):
    profiles_module.build_profiles_router(
      auth=object(),  # type: ignore[arg-type]
      profile_names_provider=lambda: (),
    )


