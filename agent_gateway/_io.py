from __future__ import annotations

import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any


log = logging.getLogger("agent_gateway.io")


def _read_json_object(path: Path) -> dict[str, Any]:
  if not path.exists():
    return {}

  try:
    payload = json.loads(path.read_text(encoding="utf-8"))
  except Exception as exc:
    log.warning("Failed to parse state file %s: %s", path, exc)
    return {}

  if isinstance(payload, dict):
    return payload

  log.warning("State file %s is not a JSON object", path)
  return {}


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  fd, tmp_path = tempfile.mkstemp(dir=path.parent, prefix=f".{path.stem}_", suffix=".tmp")
  try:
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
      json.dump(payload, handle, indent=2, sort_keys=True)
      handle.write("\n")
    os.replace(tmp_path, path)
  except BaseException:
    try:
      os.unlink(tmp_path)
    except OSError:
      pass
    raise


__all__ = [
  "_atomic_write_json",
  "_read_json_object",
]
