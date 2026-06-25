from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


PACKAGE_ROOT = Path(__file__).resolve().parents[1]


def _nav_targets(nav_items: list[Any]) -> list[str]:
  targets: list[str] = []
  for item in nav_items:
    if isinstance(item, str):
      targets.append(item)
      continue
    if not isinstance(item, dict):
      continue
    for value in item.values():
      if isinstance(value, str):
        targets.append(value)
      elif isinstance(value, list):
        targets.extend(_nav_targets(value))
  return targets


def test_mkdocs_nav_targets_exist() -> None:
  config = yaml.safe_load((PACKAGE_ROOT / "mkdocs.yml").read_text(encoding="utf-8"))
  docs_dir = PACKAGE_ROOT / config["docs_dir"]

  for target in _nav_targets(config["nav"]):
    assert (docs_dir / target).is_file(), target
