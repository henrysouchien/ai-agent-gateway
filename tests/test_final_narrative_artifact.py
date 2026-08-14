from __future__ import annotations

from pathlib import Path

import pytest

from agent_gateway.final_narrative_artifact import (
  FinalNarrativeArtifactError,
  publish_final_narrative,
  read_final_narrative,
  read_final_narrative_by_content_handle,
)
from agent_gateway.sub_agent_result_contract import (
  terminal_narrative_content_handle,
)


def test_final_narrative_round_trips_exact_unicode_and_is_idempotent(
  tmp_path: Path,
) -> None:
  text = "Opening frame.\n\nExact conclusion: café demand remains durable. 🚀"

  first = publish_final_narrative(
    workspace_dir=tmp_path,
    sub_agent_id="sub0:session-1",
    terminal_event_seq=83,
    text=text,
  )
  second = publish_final_narrative(
    workspace_dir=tmp_path,
    sub_agent_id="sub0:session-1",
    terminal_event_seq=83,
    text=text,
  )

  assert second == first
  assert first.content_chars == len(text)
  assert first.content_bytes == len(text.encode("utf-8"))
  assert first.terminal_event_seq == 83
  assert read_final_narrative(
    workspace_dir=tmp_path,
    reference=first,
  ) == text
  assert read_final_narrative_by_content_handle(
    workspace_dir=tmp_path,
    content=terminal_narrative_content_handle(first),
  ) == text


def test_content_index_resolves_identical_bytes_across_attempts(
  tmp_path: Path,
) -> None:
  text = "Same exact terminal narrative."
  first = publish_final_narrative(
    workspace_dir=tmp_path,
    sub_agent_id="child-a",
    terminal_event_seq=10,
    text=text,
  )
  second = publish_final_narrative(
    workspace_dir=tmp_path,
    sub_agent_id="child-b",
    terminal_event_seq=20,
    text=text,
  )

  assert first.artifact_id != second.artifact_id
  assert read_final_narrative_by_content_handle(
    workspace_dir=tmp_path,
    content=terminal_narrative_content_handle(second),
  ) == text


def test_final_narrative_round_trips_more_than_four_mibibytes(
  tmp_path: Path,
) -> None:
  # The 15-byte unit also exercises a multi-byte code point split across the
  # reader's 1 MiB byte boundary.
  unit = "漢字🙂café"
  unit_count = 300_000
  ending = "Exact ending: 東京 🚀"
  text = (unit * unit_count) + ending

  reference = publish_final_narrative(
    workspace_dir=tmp_path,
    sub_agent_id="sub0:large-session",
    terminal_event_seq=144,
    text=text,
  )

  assert reference.content_bytes > 4 * 1024 * 1024
  assert reference.content_bytes == (
    len(unit.encode("utf-8")) * unit_count
    + len(ending.encode("utf-8"))
  )
  assert reference.content_chars == len(text)
  assert read_final_narrative(
    workspace_dir=tmp_path,
    reference=reference,
  ) == text


def test_final_narrative_reader_rejects_payload_tampering(tmp_path: Path) -> None:
  reference = publish_final_narrative(
    workspace_dir=tmp_path,
    sub_agent_id="sub0:session-1",
    terminal_event_seq=83,
    text="Verified terminal narrative.",
  )
  payload_path = Path(tmp_path) / reference.artifact_ref
  payload_path = payload_path.parent / "payload.txt"
  payload_path.write_text("Tampered terminal narrative.", encoding="utf-8")

  with pytest.raises(FinalNarrativeArtifactError, match="content hash changed"):
    read_final_narrative(
      workspace_dir=tmp_path,
      reference=reference,
    )


def test_final_narrative_publisher_rejects_symlink_workspace(
  tmp_path: Path,
) -> None:
  workspace = tmp_path / "workspace"
  workspace.mkdir()
  alias = tmp_path / "workspace-alias"
  alias.symlink_to(workspace, target_is_directory=True)

  with pytest.raises(FinalNarrativeArtifactError, match="non-symlink"):
    publish_final_narrative(
      workspace_dir=alias,
      sub_agent_id="sub0:session-1",
      terminal_event_seq=83,
      text="Terminal narrative.",
    )
