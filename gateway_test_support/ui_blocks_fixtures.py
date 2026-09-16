"""Load packaged UI-block fixtures for development and contract tests."""

from __future__ import annotations

import json
from typing import Any

from agent_gateway.ui_blocks_contract import packaged_contract_directory


def load_ui_blocks_fixtures() -> list[dict[str, Any]]:
  fixture_directory = packaged_contract_directory() / "fixtures"
  fixture_paths = sorted(
    fixture_directory.glob("*.json"), key=lambda path: path.name.encode("utf-8")
  )
  if not fixture_paths:
    raise ValueError("UI blocks fixtures are absent")
  loaded: list[dict[str, Any]] = []
  for path in fixture_paths:
    try:
      fixture = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
      raise ValueError(f"invalid UI blocks fixture JSON: {path}") from exc
    if not isinstance(fixture, dict):
      raise ValueError(f"UI blocks fixture must be an object: {path.name}")
    if fixture.get("expectation") == "reject" and not isinstance(
      fixture.get("expected_code"), str
    ):
      raise ValueError(f"UI blocks negative fixture lacks expected_code: {path.name}")
    loaded.append(fixture)
  return loaded


__all__ = ["load_ui_blocks_fixtures"]
