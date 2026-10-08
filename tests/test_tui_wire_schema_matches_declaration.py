from __future__ import annotations

import difflib
import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
EXPORTER_PATH = ROOT / "scripts" / "export_tui_wire_schema.py"
GENERATED_PATH = ROOT / "packages" / "agent-gateway-tui" / "src" / "wire.generated.ts"


def _load_exporter():
  spec = importlib.util.spec_from_file_location("export_tui_wire_schema", EXPORTER_PATH)
  assert spec is not None and spec.loader is not None
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


def test_tui_wire_schema_matches_gateway_declaration():
  exporter = _load_exporter()
  rendered = exporter.render().encode("utf-8")
  checked_in = GENERATED_PATH.read_bytes()
  if rendered != checked_in:
    diff = "".join(
      difflib.unified_diff(
        checked_in.decode("utf-8").splitlines(keepends=True),
        rendered.decode("utf-8").splitlines(keepends=True),
        fromfile="wire.generated.ts (checked in)",
        tofile="wire.generated.ts (declared)",
      )
    )
    raise AssertionError(
      f"{diff}\nwire.generated.ts drifted from V1_WIRE_EVENT_TYPES/V1_FIELD_PROJECTION; "
      "run: npm --prefix packages/agent-gateway-tui run gen:wire"
    )
