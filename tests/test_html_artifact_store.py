from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[3]
PKG_DIR = ROOT / "packages" / "agent-gateway"
if str(PKG_DIR) not in sys.path:
  sys.path.insert(0, str(PKG_DIR))

import agent_gateway.html_artifact_store as html_store
from agent_gateway.artifact_sidecar_index import get_artifact_sidecar_index_row
from agent_gateway.html_artifact_store import (
  list_html_artifacts,
  read_html_artifact_content,
  read_html_artifact_sidecar,
  validate_html_artifact_content,
  write_html_artifact,
)
from schema.html_artifact import HtmlArtifact, StaticExports


def test_html_artifact_store_round_trips_sidecar_and_content(tmp_path: Path) -> None:
  artifact = _artifact("html-artifact-1", ticker="PCTY")

  write_html_artifact(
    workspace_dir=tmp_path,
    artifact=artifact,
    html_content="<section><h1>PCTY</h1></section>",
  )

  assert read_html_artifact_sidecar(tmp_path, "html-artifact-1") == artifact
  assert read_html_artifact_content(tmp_path, "html-artifact-1") == "<section><h1>PCTY</h1></section>"


def test_html_artifact_store_registers_sidecar_index_row(tmp_path: Path) -> None:
  workspace = tmp_path / "users" / "alice" / "workspace"
  artifact = _artifact("html-artifact-1", ticker="PCTY", purpose="report")

  write_html_artifact(
    workspace_dir=workspace,
    artifact=artifact,
    html_content="<section><h1>PCTY</h1></section>",
  )

  row = get_artifact_sidecar_index_row(
    workspace_dir=workspace,
    artifact_kind="html",
    artifact_id="html-artifact-1",
    user_id="alice",
  )
  assert row is not None
  assert row["artifact_ref"] == "artifacts/_html/html-artifact-1.json"
  assert row["payload_ref"] == "artifacts/_html/html-artifact-1.html"
  assert row["scope"] == "ticker"
  assert row["scope_label"] is None
  assert row["ticker"] == "PCTY"
  assert row["skill"] == "historical-coincidences"
  assert row["purpose"] == "report"
  assert row["contract_name"] == "HtmlArtifact"
  assert row["classification_source"] == "legacy_default"
  assert row["index_version"] == 1
  assert row["last_seen_ts"]
  assert row["stale_ts"] is None
  assert row["last_error"] is None
  assert row["content_hash"]
  assert get_artifact_sidecar_index_row(
    workspace_dir=workspace,
    artifact_kind="html",
    artifact_id="html-artifact-1",
  ) == row


def test_html_artifact_store_keeps_sidecar_when_index_registration_fails(
  tmp_path: Path,
  monkeypatch: pytest.MonkeyPatch,
  caplog: pytest.LogCaptureFixture,
) -> None:
  artifact = _artifact("index-failure", ticker="PCTY")

  def _raise(*args, **kwargs):
    raise RuntimeError("index unavailable")

  monkeypatch.setattr(html_store, "register_html_artifact_sidecar", _raise)

  with caplog.at_level(logging.WARNING, logger="agent_gateway.html_artifact_store"):
    write_html_artifact(
      workspace_dir=tmp_path,
      artifact=artifact,
      html_content="<section><h1>PCTY</h1></section>",
    )

  assert read_html_artifact_sidecar(tmp_path, "index-failure") == artifact
  assert read_html_artifact_content(tmp_path, "index-failure") == "<section><h1>PCTY</h1></section>"
  assert "artifact_index_failure" in caplog.messages


def test_html_artifact_store_rejects_mismatched_index_user_without_failing_sidecar(
  tmp_path: Path,
  caplog: pytest.LogCaptureFixture,
) -> None:
  workspace = tmp_path / "users" / "alice" / "workspace"
  artifact = _artifact("wrong-user", ticker="PCTY")

  with caplog.at_level(logging.WARNING, logger="agent_gateway.html_artifact_store"):
    write_html_artifact(
      workspace_dir=workspace,
      artifact=artifact,
      html_content="<section><h1>PCTY</h1></section>",
      user_id="bob",
    )

  assert read_html_artifact_sidecar(workspace, "wrong-user") == artifact
  assert get_artifact_sidecar_index_row(
    workspace_dir=workspace,
    artifact_kind="html",
    artifact_id="wrong-user",
  ) is None
  assert "artifact_index_failure" in caplog.messages


def test_html_artifact_store_lists_newest_first_with_filters(tmp_path: Path) -> None:
  old_artifact = _artifact("old-artifact", ticker="PCTY", purpose="exploration")
  new_artifact = _artifact("new-artifact", ticker=None, purpose="report")
  other_artifact = _artifact("other-artifact", ticker="MSFT", purpose="report")

  for artifact in [old_artifact, new_artifact, other_artifact]:
    write_html_artifact(
      workspace_dir=tmp_path,
      artifact=artifact,
      html_content=f"<p>{artifact.artifact_id}</p>",
    )

  _set_sidecar_mtime(tmp_path, old_artifact.artifact_id, 100)
  _set_sidecar_mtime(tmp_path, new_artifact.artifact_id, 300)
  _set_sidecar_mtime(tmp_path, other_artifact.artifact_id, 200)

  assert [artifact.artifact_id for artifact in list_html_artifacts(tmp_path)] == [
    "new-artifact",
    "other-artifact",
    "old-artifact",
  ]
  assert [artifact.artifact_id for artifact in list_html_artifacts(tmp_path, ticker="PCTY")] == [
    "old-artifact"
  ]
  assert [artifact.artifact_id for artifact in list_html_artifacts(tmp_path, purpose="report")] == [
    "new-artifact",
    "other-artifact",
  ]
  assert [artifact.artifact_id for artifact in list_html_artifacts(tmp_path, since=250)] == [
    "new-artifact"
  ]
  assert [artifact.artifact_id for artifact in list_html_artifacts(tmp_path, limit=1)] == [
    "new-artifact"
  ]


def test_html_artifact_store_rejects_unsafe_artifact_ids_before_writing(tmp_path: Path) -> None:
  artifact = _artifact("../escape", ticker="PCTY")

  try:
    write_html_artifact(workspace_dir=tmp_path, artifact=artifact, html_content="<p>bad</p>")
  except ValueError as exc:
    assert "invalid html artifact_id" in str(exc)
  else:
    raise AssertionError("unsafe artifact id should raise")

  assert not (tmp_path / "artifacts").exists()


def test_html_artifact_store_rejects_unsafe_html_before_writing(tmp_path: Path) -> None:
  artifact = _artifact("unsafe-html", ticker="PCTY")

  for html in (
    "<script>alert(1)</script>",
    "<section style='color:red'>bad</section>",
    "<a href='javascript:alert(1)'>bad</a>",
    "<a href='java\nscript:alert(1)'>bad</a>",
    "<img onerror='alert(1)' src='x'>",
  ):
    try:
      write_html_artifact(workspace_dir=tmp_path, artifact=artifact, html_content=html)
    except ValueError as exc:
      assert "forbidden" in str(exc)
    else:
      raise AssertionError(f"unsafe html should raise: {html}")

  assert not (tmp_path / "artifacts").exists()


def test_html_artifact_store_rejects_empty_or_non_string_html_before_writing(tmp_path: Path) -> None:
  artifact = _artifact("empty-html", ticker="PCTY")

  for html in ("", " \n", None):
    try:
      write_html_artifact(workspace_dir=tmp_path, artifact=artifact, html_content=html)
    except ValueError as exc:
      assert "non-empty string" in str(exc)
    else:
      raise AssertionError(f"empty HTML should raise: {html!r}")

  assert not (tmp_path / "artifacts").exists()


def test_html_artifact_validation_allows_safe_semantic_fragments() -> None:
  validate_html_artifact_content(
    "<section class='artifact-claim'><h2>Claim</h2><p>Evidence-backed view.</p></section>"
  )


def test_html_artifact_store_rejects_symlink_escape(tmp_path: Path) -> None:
  workspace = tmp_path / "workspace"
  outside = tmp_path / "outside"
  (workspace / "artifacts").mkdir(parents=True)
  outside.mkdir()
  (workspace / "artifacts" / "_html").symlink_to(outside, target_is_directory=True)

  try:
    write_html_artifact(
      workspace_dir=workspace,
      artifact=_artifact("symlink-escape", ticker="PCTY"),
      html_content="<p>safe</p>",
    )
  except ValueError as exc:
    assert "escapes workspace" in str(exc)
  else:
    raise AssertionError("symlinked html artifact directory should raise")

  assert list(outside.iterdir()) == []


def test_html_artifact_store_list_ignores_symlinked_sidecar_escape(tmp_path: Path) -> None:
  workspace = tmp_path / "workspace"
  outside = tmp_path / "outside"
  outside.mkdir()
  write_html_artifact(
    workspace_dir=workspace,
    artifact=_artifact("safe-artifact", ticker="PCTY"),
    html_content="<p>safe</p>",
  )
  outside_sidecar = outside / "leaked.json"
  outside_sidecar.write_text(_artifact("leaked", ticker="MSFT").model_dump_json(), encoding="utf-8")
  (workspace / "artifacts" / "_html" / "leaked.json").symlink_to(outside_sidecar)

  assert [artifact.artifact_id for artifact in list_html_artifacts(workspace)] == ["safe-artifact"]


def test_html_artifact_store_missing_content_is_soft_absent(tmp_path: Path) -> None:
  artifact = _artifact("sidecar-only", ticker=None)
  directory = tmp_path / "artifacts" / "_html"
  directory.mkdir(parents=True)
  (directory / "sidecar-only.json").write_text(artifact.model_dump_json(), encoding="utf-8")

  assert read_html_artifact_sidecar(tmp_path, "sidecar-only") == artifact
  assert read_html_artifact_content(tmp_path, "sidecar-only") is None


def test_html_artifact_store_ignores_partial_temp_writes(tmp_path: Path) -> None:
  directory = tmp_path / "artifacts" / "_html"
  directory.mkdir(parents=True)
  (directory / "partial.html.tmp").write_text("<p>partial</p>", encoding="utf-8")

  assert list_html_artifacts(tmp_path) == []


def test_html_artifact_content_policy_rejects_renderer_invalid_html() -> None:
  invalid_cases = [
    ("<section>Risk bridge</section><style>body{color:red}</style>", "forbidden HTML tag: style"),
    ("<section style='color:red'>Risk bridge</section>", "forbidden HTML attribute: style"),
    ("<button onclick='alert(1)'>Risk bridge</button>", "forbidden HTML attribute: onclick"),
    ("<a href=' java&#x0A;script:alert(1)'>Risk bridge</a>", "forbidden javascript URL"),
    ("<svg><a xlink:href='java\tscript:alert(1)'>Risk bridge</a></svg>", "forbidden javascript URL"),
  ]

  for html, expected in invalid_cases:
    try:
      validate_html_artifact_content(html)
    except ValueError as exc:
      assert expected in str(exc)
    else:
      raise AssertionError(f"invalid HTML should raise: {html}")

  validate_html_artifact_content(
    "<main><p class='artifact-claim'>Risk bridge</p><section class='artifact-exhibit'>Exhibit</section></main>"
  )


def _artifact(
  artifact_id: str,
  *,
  ticker: str | None,
  purpose: str = "exploration",
) -> HtmlArtifact:
  return HtmlArtifact(
    artifact_id=artifact_id,
    title=f"{artifact_id} title",
    purpose=purpose,
    content_ref=f"{artifact_id}.html",
    summary=f"{artifact_id} summary",
    ticker=ticker,
    session_id=None,
    source_skill="historical-coincidences",
    sources=[],
    exports=StaticExports(
      copy_as_prompt="Prompt export",
      copy_as_markdown=None,
      copy_as_json={"artifact_id": artifact_id},
    ),
    ts="2026-06-01T12:00:00+00:00",
  )


def _set_sidecar_mtime(workspace_dir: Path, artifact_id: str, mtime: int) -> None:
  path = workspace_dir / "artifacts" / "_html" / f"{artifact_id}.json"
  os.utime(path, (mtime, mtime))
