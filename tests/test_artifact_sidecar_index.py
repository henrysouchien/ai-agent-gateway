from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
PKG_DIR = ROOT / "packages" / "agent-gateway"
if str(PKG_DIR) not in sys.path:
  sys.path.insert(0, str(PKG_DIR))

from agent_gateway.artifact_sidecar_index import (  # noqa: E402
  get_artifact_sidecar_index_row,
  get_artifact_sidecar_index_row_by_ref,
  register_skill_artifact_sidecar,
)


def test_register_skill_artifact_sidecar_indexes_ticker_metadata(tmp_path: Path) -> None:
  workspace = tmp_path / "users" / "alice" / "workspace"
  sidecar_path = workspace / "artifacts" / "PCTY" / "model-review" / "2026-06-18T120000.000-run-a.json"
  payload = {
    "artifact_id": "2026-06-18T120000.000-run-a",
    "ticker": "PCTY",
    "skill": "model-review",
    "contract_name": "ModelReviewArtifact",
    "created_at": "2026-06-18T12:00:00+00:00",
    "binary_artifact_path": "artifacts/PCTY/model-review/2026-06-18T120000.000-run-a.md",
    "research_file_id": 42,
    "control_run_id": "bg_123",
    "origin_kind": "product",
    "visibility": "default",
    "origin_ref": {"kind": "research_file", "id": 42},
  }
  _write_json(sidecar_path, payload)

  register_skill_artifact_sidecar(workspace_dir=workspace, sidecar_path=sidecar_path)

  row = get_artifact_sidecar_index_row(
    workspace_dir=workspace,
    artifact_kind="skill_artifact",
    artifact_id="2026-06-18T120000.000-run-a",
    user_id="alice",
  )
  assert row is not None
  assert row["artifact_ref"] == "artifacts/PCTY/model-review/2026-06-18T120000.000-run-a.json"
  assert row["payload_ref"] == "artifacts/PCTY/model-review/2026-06-18T120000.000-run-a.md"
  assert row["scope"] == "ticker"
  assert row["ticker"] == "PCTY"
  assert row["skill"] == "model-review"
  assert row["contract_name"] == "ModelReviewArtifact"
  assert row["research_file_id"] == 42
  assert row["control_run_id"] == "bg_123"
  assert row["origin_kind"] == "product"
  assert row["visibility"] == "default"
  assert row["classification_source"] == "sidecar"
  assert row["origin_ref"] == '{"id":42,"kind":"research_file"}'
  assert row["created_ts"] == "2026-06-18T12:00:00+00:00"
  assert row["content_hash"]


def test_register_skill_artifact_sidecar_indexes_portfolio_metadata(tmp_path: Path) -> None:
  workspace = tmp_path / "users" / "alice" / "workspace"
  sidecar_path = workspace / "artifacts" / "_portfolio" / "risk-review" / "2026-06-18T120000.000-run-a.json"
  _write_json(
    sidecar_path,
    {
      "artifact_id": "2026-06-18T120000.000-run-a",
      "skill": "risk-review",
      "contract_name": "RiskReviewArtifact",
      "research_file_id": 99,
    },
  )

  register_skill_artifact_sidecar(workspace_dir=workspace, sidecar_path=sidecar_path)

  row = get_artifact_sidecar_index_row_by_ref(
    workspace_dir=workspace,
    artifact_kind="skill_artifact",
    artifact_ref="artifacts/_portfolio/risk-review/2026-06-18T120000.000-run-a.json",
  )
  assert row is not None
  assert row["scope"] == "portfolio"
  assert row["ticker"] is None
  assert row["skill"] == "risk-review"
  assert row["classification_source"] == "unresolved_research_file"


def _write_json(path: Path, payload: dict[str, object]) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
