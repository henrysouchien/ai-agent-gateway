from __future__ import annotations

from pathlib import Path

import pytest

from agent_gateway.artifact_paths import (
  ArtifactPathError,
  artifact_json_path_for_request,
  artifact_json_paths_for_request,
  ticker_artifact_paths_for_request,
)


def test_share_class_tickers_normalize_to_letter_only_artifact_paths(
  tmp_path: Path,
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  monkeypatch.setenv("USER_DATA_DIR", str(tmp_path / "data"))

  artifact = artifact_json_path_for_request(
    "alice",
    ticker="BRK-B",
    skill="thesis-review",
    artifact_id="2026-06-30T000000.000-run-a",
  )

  assert artifact.ticker == "BRKB"
  assert artifact.path == (
    tmp_path
    / "data"
    / "users"
    / "alice"
    / "workspace"
    / "artifacts"
    / "BRKB"
    / "thesis-review"
    / "2026-06-30T000000.000-run-a.json"
  ).resolve()


@pytest.mark.parametrize("ticker,normalized", [("HPS-A", "HPSA"), ("MOG-A", "MOGA")])
def test_hyphen_share_class_ticker_artifact_lists_are_safe_when_empty(
  tmp_path: Path,
  monkeypatch: pytest.MonkeyPatch,
  ticker: str,
  normalized: str,
) -> None:
  monkeypatch.setenv("USER_DATA_DIR", str(tmp_path / "data"))

  artifacts = artifact_json_paths_for_request("alice", ticker=ticker, skill="earnings-scenarios")
  by_skill = ticker_artifact_paths_for_request("alice", ticker=ticker)

  assert artifacts == []
  assert by_skill == {}

  target = artifact_json_path_for_request(
    "alice",
    ticker=ticker,
    skill="earnings-scenarios",
    artifact_id="2026-06-30T000000.000-run-a",
  )
  assert target.ticker == normalized


@pytest.mark.parametrize("ticker", ["EFC-PC", "PPL-PA", "BRK-", "-A", ".A", "../MSFT", "MS/FT", "MS%2FFT"])
def test_unsafe_or_non_common_equity_tickers_stay_rejected(
  tmp_path: Path,
  monkeypatch: pytest.MonkeyPatch,
  ticker: str,
) -> None:
  monkeypatch.setenv("USER_DATA_DIR", str(tmp_path / "data"))

  with pytest.raises(ArtifactPathError):
    artifact_json_path_for_request(
      "alice",
      ticker=ticker,
      skill="earnings-scenarios",
      artifact_id="2026-06-30T000000.000-run-a",
    )
