from pathlib import Path
from types import SimpleNamespace

from agent_gateway.runtime_spill import build_spill_sink, sweep_direct_spill_roots


def test_direct_spill_root_readme_survives_stale_run_sweep(tmp_path: Path) -> None:
  workspace = tmp_path / "workspace"
  workspace.mkdir()
  sink = build_spill_sink(
    SimpleNamespace(workspace=workspace),
    run_id="stale-run",
    available_tool_names=set(),
  )
  assert sink is not None

  spill_dir = Path(sink())
  lease = getattr(sink, "_autonomous_spill_lease")
  assert lease is not None
  lease.close()

  runs_root = workspace / ".agent-runs"
  readme = runs_root / "README.md"
  assert readme.is_file()

  freshness = spill_dir.parent / ".spill_fresh"
  removed = sweep_direct_spill_roots(
    workspace,
    ttl_hours=1.0,
    now=freshness.stat().st_mtime + 3601.0,
  )

  assert removed == 1
  assert not spill_dir.parent.exists()
  assert readme.is_file()
