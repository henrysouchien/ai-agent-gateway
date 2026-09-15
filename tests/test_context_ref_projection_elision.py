from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
PKG_DIR = ROOT / "packages" / "agent-gateway"
if str(ROOT) not in sys.path:
  sys.path.insert(0, str(ROOT))
if str(PKG_DIR) not in sys.path:
  sys.path.insert(0, str(PKG_DIR))

from agent_gateway.tool_result_compaction import (  # noqa: E402
  _ELIDED_CHARS_MARKER,
  truncate_model_tool_result_content,
)
from schema.business_model_revision_ref import BusinessModelRevisionRef  # noqa: E402
from schema.forecast_assumptions_context import (  # noqa: E402
  ForecastAssumptionsContextBinding,
  encode_forecast_assumptions_context_ref,
  forecast_assumptions_context_content_digest,
)
from schema.forecast_control_profile import ForecastControlProfileRef  # noqa: E402
from schema.historical_foundation import (  # noqa: E402
  HistoricalFactSnapshotRef,
  HistoricalFoundationRef,
)


def _forecast_assumptions_context_ref() -> str:
  sha = "a" * 64
  revision = BusinessModelRevisionRef(
    business_model_id="PCTY_business_model",
    revision="pcty-bm-v1",
    content_sha256=sha,
  )
  foundation = HistoricalFoundationRef(
    artifact_id="hff_" + "b" * 32,
    content_sha256="b" * 64,
  )
  fact_snapshot = HistoricalFactSnapshotRef(
    snapshot_id="hfs_" + "d" * 32,
    content_sha256="d" * 64,
  )
  profile = ForecastControlProfileRef(
    profile_id="sia-generic.nonvaluation-forecast-controls",
    declared_version="1",
    content_sha256="f" * 64,
  )
  digest = forecast_assumptions_context_content_digest(
    user_id="henry",
    research_file_id=1,
    ticker="PCTY",
    selected_business_model_revision_ref=revision,
    business_model_selection_epoch=1,
    business_model_stage_receipt_digest="c" * 64,
    historical_foundation_ref=foundation,
    historical_foundation_stage_receipt_digest="e" * 64,
    historical_fact_snapshot_ref=fact_snapshot,
    forecast_control_profile_ref=profile,
    forecast_control_denominator_sha256="1" * 64,
  )
  binding = ForecastAssumptionsContextBinding(
    user_id="henry",
    research_file_id=1,
    ticker="PCTY",
    selected_business_model_revision_ref=revision,
    business_model_selection_epoch=1,
    business_model_stage_receipt_digest="c" * 64,
    historical_foundation_ref=foundation,
    historical_foundation_stage_receipt_digest="e" * 64,
    historical_fact_snapshot_ref=fact_snapshot,
    forecast_control_profile_ref=profile,
    forecast_control_denominator_sha256="1" * 64,
    context_content_sha256=digest,
  )
  return encode_forecast_assumptions_context_ref(binding)


def test_context_ref_locator_survives_projection_elision_intact() -> None:
  locator = _forecast_assumptions_context_ref()
  content = json.dumps(
    {
      "status": "success",
      "context_ref": locator,
      "payload": "x" * 20_000,
    }
  )
  truncated, was_truncated = truncate_model_tool_result_content(
    content,
    tool_name="get_forecast_assumptions_context",
    max_chars=4_000,
  )
  assert was_truncated is True
  assert len(truncated) <= 4_000
  projection = json.loads(truncated)["content_projection"]
  assert projection["context_ref"] == locator
  assert _ELIDED_CHARS_MARKER not in projection["context_ref"]
  assert _ELIDED_CHARS_MARKER in str(projection["payload"])
