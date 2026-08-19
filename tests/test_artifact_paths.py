from __future__ import annotations

from pathlib import Path

import pytest

from agent_gateway.artifact_paths import (
  ArtifactPathError,
  canonicalize_ticker,
  artifact_json_path_for_request,
  artifact_json_paths_for_request,
  ticker_artifact_paths_for_request,
)


@pytest.mark.parametrize(
  "ticker,expected",
  [(" taee11.sa ", "TAEE11"), ("0700.HK", "0700"), ("BRK-B", "BRKB"), ("AAPL", "AAPL")],
)
def test_lh16_explicit_ticker_canonicalization_vectors(ticker: str, expected: str) -> None:
  assert canonicalize_ticker(ticker) == expected


@pytest.mark.parametrize("ticker", ["A..B", "EFC-PC", ""])
def test_lh16_explicit_ticker_canonicalization_rejects_invalid_shapes(ticker: str) -> None:
  with pytest.raises(ValueError):
    canonicalize_ticker(ticker)


@pytest.mark.parametrize("ticker", [None, 123, True])
def test_explicit_ticker_canonicalization_rejects_non_strings(
  ticker: object,
) -> None:
  with pytest.raises(ValueError, match="must be a string|required"):
    canonicalize_ticker(ticker)


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


def test_non_us_numeric_tickers_resolve_for_read_and_list(
  tmp_path: Path,
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  """LH-12 reader symmetry: artifacts persisted under non-US tickers (digits,

  exchange-suffix dots) must stay readable/listable through the public request
  paths — the writer side (api/research/artifact_paths.py) accepts them
  verbatim under the same extended rule."""
  monkeypatch.setenv("USER_DATA_DIR", str(tmp_path / "data"))

  artifact = artifact_json_path_for_request(
    "alice",
    ticker="TAEE11",
    skill="sniff-test",
    artifact_id="2026-07-09T233509.044-run-a",
  )
  assert artifact.ticker == "TAEE11"
  assert artifact.path.parent.parts[-3:] == ("artifacts", "TAEE11", "sniff-test")

  # Persist at the resolved path, then list + read back through request paths.
  artifact.path.parent.mkdir(parents=True, exist_ok=True)
  artifact.path.write_text('{"skill": "sniff-test", "ticker": "TAEE11"}', encoding="utf-8")

  listed = artifact_json_paths_for_request("alice", ticker="TAEE11", skill="sniff-test")
  assert [p.path for p in listed] == [artifact.path]

  by_skill = ticker_artifact_paths_for_request("alice", ticker="TAEE11")
  assert "sniff-test" in by_skill


@pytest.mark.parametrize("ticker,resolved", [("0700", "0700"), ("PETR4", "PETR4"), ("TAEE11.SA", "TAEE11.SA")])
def test_extended_tickers_accepted_verbatim(
  tmp_path: Path,
  monkeypatch: pytest.MonkeyPatch,
  ticker: str,
  resolved: str,
) -> None:
  monkeypatch.setenv("USER_DATA_DIR", str(tmp_path / "data"))

  artifact = artifact_json_path_for_request(
    "alice",
    ticker=ticker,
    skill="sniff-test",
    artifact_id="2026-07-09T233509.044-run-a",
  )
  assert artifact.ticker == resolved
