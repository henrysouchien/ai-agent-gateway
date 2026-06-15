from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .registry import registry_description


DESCRIPTION_PATH = Path(__file__).with_name("registry_description.json")


def build_description() -> dict[str, Any]:
  return registry_description()


def render_description() -> str:
  return json.dumps(build_description(), indent=2, sort_keys=True) + "\n"


def write_description(path: Path = DESCRIPTION_PATH) -> None:
  path.write_text(render_description(), encoding="utf-8")


def main() -> None:
  write_description()


if __name__ == "__main__":
  main()
