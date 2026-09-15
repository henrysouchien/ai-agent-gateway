from __future__ import annotations

import ast
from collections.abc import Iterator
from dataclasses import fields
import json
import os
from pathlib import Path
import subprocess
import sys
from types import TracebackType
from typing import BinaryIO, Literal

import pytest

from agent_gateway.control_skill_catalog import (
  ControlSkillCatalog,
  ControlSkillDetail,
  ControlSkillSummary,
  ControlSkillUnavailableError,
)
from agent_gateway.directory_control_skill_catalog import (
  DirectoryControlSkillCatalog,
)
from agent_gateway.skills import SkillLoader


ROOT = Path(__file__).resolve().parents[1]

_ABSOLUTE_IMPORTS = {
  "__future__": frozenset({"annotations"}),
  "collections.abc": frozenset({"Mapping"}),
  "dataclasses": frozenset({"dataclass"}),
  "pathlib": frozenset({"Path"}),
  "typing": frozenset({"Any"}),
}
_DIRECT_IMPORTS = frozenset({"logging", "os", "stat", "yaml"})
_RELATIVE_IMPORTS = {
  "control_skill_catalog": frozenset({
    "ControlSkillDetail",
    "ControlSkillSummary",
    "ControlSkillUnavailableError",
    "is_control_skill_name",
  }),
  "skills": frozenset({"SkillLoader", "SkillProfile", "resolve_blocks"}),
}


def _assert_directory_import_boundary(source: str) -> None:
  tree = ast.parse(source)
  for node in ast.walk(tree):
    if isinstance(node, ast.Import):
      assert len(node.names) == 1
      alias = node.names[0]
      assert alias.asname is None
      assert alias.name in _DIRECT_IMPORTS
      continue
    if not isinstance(node, ast.ImportFrom):
      continue
    allowed = (
      _ABSOLUTE_IMPORTS.get(node.module or "")
      if node.level == 0
      else _RELATIVE_IMPORTS.get(node.module or "")
      if node.level == 1
      else None
    )
    assert allowed is not None
    for alias in node.names:
      assert alias.asname is None
      assert alias.name in allowed


def _write_skill(
  root: Path,
  name: str,
  *,
  frontmatter: str = "",
  body: str = "Methodology.",
) -> Path:
  root.mkdir(parents=True, exist_ok=True)
  path = root / f"{name}.md"
  path.write_text(
    f"---\n{frontmatter.strip()}\n---\n{body}",
    encoding="utf-8",
  )
  return path


def _rich_frontmatter(*, description: str = "Rich description.") -> str:
  return f"""
name: rich-alias
description: {description}
version: '2.1'
scope: portfolio
agent_callable: true
agent_description: Agent-facing detail.
resumable: true
max_turns: -4
persist_state: true
metadata:
  label:
  display_label: Must Not Win
  typed_contract: platform:external-result@7
  max_budget_usd: -3.5
  required_context: [portfolio, ticker, ticker, '  ']
  requires_portfolio_context: 'yes'
  modes: ''
  compatible_modes: [must-not-win]
  tier_availability: [paid, paid, ' enterprise ']
semantic_metadata:
  allowed_profiles: [advisor, ' analyst ', advisor]
  allowed_effects: [read, state_write, external_write]
  approval_constraints:
    - runtime_policy
    - human_review_before_apply
    - explicit_user_approval
  output_contracts:
    - owner: ai
      contract_name: analysis-result
      schema_version: 2
    - owner: platform
      contract_name: final-result
      schema_version: '1'
  credential_requirements: [market_data, ' portfolio_connection ']
  scheduling:
    eligibility: eligible
"""


def test_directory_catalog_projects_complete_27_field_external_table(
  tmp_path: Path,
) -> None:
  root = tmp_path / "external-skills"
  _write_skill(
    root,
    "filename-alias",
    frontmatter=_rich_frontmatter(),
    body="Before {{COMMON}} After",
  )
  blocks = root / "_blocks"
  blocks.mkdir()
  (blocks / "common.md").write_text("Shared block.", encoding="utf-8")

  catalog = DirectoryControlSkillCatalog(root)
  assert isinstance(catalog, ControlSkillCatalog)
  listing = catalog.list_skills()
  assert type(listing) is tuple
  assert len(listing) == 1
  summary = listing[0]
  assert type(summary) is ControlSkillSummary
  assert len(fields(summary)) == 27
  assert summary.name == "rich-alias"
  assert summary.label == "Rich Alias"
  assert summary.description == "Rich description."
  assert summary.agent_description == "Agent-facing detail."
  assert summary.version == "2.1"
  assert summary.scope == "portfolio"
  assert summary.requires_portfolio_context is True
  assert summary.required_context == ("portfolio", "ticker")
  assert summary.agent_callable is True
  assert summary.resumable is True
  assert summary.max_turns == -4
  assert summary.max_budget_usd == -3.5
  assert type(summary.max_budget_usd) is float
  assert summary.persist_state is True
  assert summary.typed_contract == "platform:external-result@7"
  assert summary.catalog is True
  assert summary.profiles == ("advisor", "analyst")
  assert summary.modes == ()
  assert summary.outputs == (
    "ai:analysis-result@2",
    "platform:final-result@1",
  )
  assert summary.action_class == "external"
  assert summary.approval_policy == "explicit_user_approval"
  assert summary.tier_availability == ("paid", "enterprise")
  assert summary.credential_requirements == (
    "market_data",
    "portfolio_connection",
  )
  assert summary.schedule_eligible is True
  assert summary.can_launch is True
  assert summary.can_schedule is True
  assert summary.blocked_reason is None
  assert summary.path == (root / "filename-alias.md").as_posix()
  assert not hasattr(summary, "body")

  detail = catalog.resolve_skill("filename-alias")
  assert type(detail) is ControlSkillDetail
  assert detail.name == "rich-alias"
  assert detail.body == "Before Shared block. After"
  for field in fields(ControlSkillSummary):
    assert getattr(detail, field.name) == getattr(summary, field.name)
  assert detail is not summary


def test_first_present_empty_aliases_suppress_later_aliases(tmp_path: Path) -> None:
  root = tmp_path / "skills"
  _write_skill(
    root,
    "alias-skill",
    frontmatter="""
name: alias-skill
metadata:
  label: ''
  display_label: Later Label
  modes:
  compatible_modes: [later-mode]
  tier_availability: ''
  tier: [later-tier]
""",
    body="",
  )

  summary = DirectoryControlSkillCatalog(root).list_skills()[0]
  assert summary.label == "Alias Skill"
  assert summary.modes == ()
  assert summary.tier_availability == ()
  assert summary.description == ""
  assert summary.version == ""
  assert DirectoryControlSkillCatalog(root).resolve_skill("alias-skill").body == ""


def test_missing_semantic_containers_use_shallow_defaults(tmp_path: Path) -> None:
  root = tmp_path / "skills"
  _write_skill(root, "minimal-skill", frontmatter="name: minimal-skill")

  summary = DirectoryControlSkillCatalog(root).list_skills()[0]
  assert summary.profiles == ()
  assert summary.outputs == ()
  assert summary.credential_requirements == ()
  assert summary.action_class == "read_only"
  assert summary.approval_policy == "runtime_policy"
  assert summary.schedule_eligible is False
  assert summary.can_launch is False
  assert summary.can_schedule is False


@pytest.mark.parametrize("nested_catalog", ["false", "[malformed]"])
def test_nested_metadata_catalog_is_not_visibility_authority(
  tmp_path: Path,
  nested_catalog: str,
) -> None:
  root = tmp_path / "skills"
  _write_skill(
    root,
    "nested-catalog",
    frontmatter=(
      "name: nested-catalog\n"
      "metadata:\n"
      f"  catalog: {nested_catalog}"
    ),
    body="Nested catalog is descriptive only.",
  )

  catalog = DirectoryControlSkillCatalog(root)
  assert tuple(summary.name for summary in catalog.list_skills()) == (
    "nested-catalog",
  )
  assert catalog.resolve_skill("nested-catalog").body == (
    "Nested catalog is descriptive only."
  )


def test_hidden_skill_skips_loader_and_detail_matches_absent(
  tmp_path: Path,
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  root = tmp_path / "skills"
  _write_skill(
    root,
    "hidden-skill",
    frontmatter="catalog: 'off'\nmax_turns: not-an-int",
  )

  class RefusingLoader:
    def __init__(self, _root: object) -> None:
      raise AssertionError("hidden skill reached SkillLoader")

  monkeypatch.setattr(
    "agent_gateway.directory_control_skill_catalog.SkillLoader",
    RefusingLoader,
  )
  catalog = DirectoryControlSkillCatalog(root)
  assert catalog.list_skills() == ()
  with pytest.raises(ControlSkillUnavailableError) as hidden:
    catalog.resolve_skill("hidden-skill")
  with pytest.raises(ControlSkillUnavailableError) as absent:
    catalog.resolve_skill("absent-skill")
  assert hidden.value.code == absent.value.code == "unknown"
  assert str(hidden.value) == str(absent.value)
  for error in (hidden.value, absent.value):
    assert error.__cause__ is None
    assert error.__context__ is None


@pytest.mark.parametrize("operation", ["list", "detail"])
@pytest.mark.parametrize(
  "newline",
  [b"\r\n", b"\r"],
  ids=("crlf", "bare-cr"),
)
def test_hidden_source_stops_at_binary_frontmatter_delimiter(
  tmp_path: Path,
  monkeypatch: pytest.MonkeyPatch,
  operation: str,
  newline: bytes,
) -> None:
  root = tmp_path / "skills"
  root.mkdir()
  path = root / "hidden-binary.md"
  header = newline.join((
    b"---",
    b"name: hidden-binary",
    b"catalog: false",
    b"---",
  )) + newline
  path.write_bytes(header + (b"x" * 262_144) + b"\xff")
  real_open = Path.open
  header_bytes: list[bytes] = []
  body_reads: list[bool] = []
  binary_opens: list[tuple[tuple[object, ...], dict[str, object]]] = []

  class FrontmatterOnlyHandle:
    def __init__(self, handle: BinaryIO) -> None:
      self._handle = handle

    def __enter__(self) -> FrontmatterOnlyHandle:
      self._handle.__enter__()
      return self

    def __exit__(
      self,
      exc_type: type[BaseException] | None,
      exc_value: BaseException | None,
      traceback: TracebackType | None,
    ) -> None:
      self._handle.__exit__(exc_type, exc_value, traceback)

    def fileno(self) -> int:
      return self._handle.fileno()

    def read(self, size: int = -1) -> bytes:
      if size == 1:
        byte = self._handle.read(size)
        header_bytes.append(byte)
        return byte
      body_reads.append(True)
      raise AssertionError("hidden body bytes must not be bulk-read")

  def guarded_open(
    candidate: Path,
    *args: Literal["rb"],
    buffering: int = -1,
  ) -> FrontmatterOnlyHandle | BinaryIO:
    handle = real_open(candidate, "rb", buffering=buffering)
    if candidate == path:
      binary_opens.append((args, {"buffering": buffering}))
      return FrontmatterOnlyHandle(handle)
    return handle

  monkeypatch.setattr(Path, "open", guarded_open)
  catalog = DirectoryControlSkillCatalog(root)

  if operation == "list":
    assert catalog.list_skills() == ()
  else:
    with pytest.raises(ControlSkillUnavailableError) as error:
      catalog.resolve_skill("hidden-binary")
    assert error.value.code == "unknown"
    assert error.value.__cause__ is None
    assert error.value.__context__ is None
  assert body_reads == []
  assert binary_opens == [(("rb",), {"buffering": 0})]
  expected_prefix = header[:-1] if newline == b"\r\n" else header
  assert b"".join(header_bytes) == expected_prefix


@pytest.mark.parametrize(
  "outside_contents",
  [
    "---\ncatalog: false\n---\nHidden",
    "---\ncatalog: true\nname: escaped-skill\n---\nVisible",
    "---\ncatalog: [\n---\nMalformed",
  ],
  ids=("hidden", "visible", "malformed"),
)
def test_escape_symlink_is_absent_before_any_visibility_read(
  tmp_path: Path,
  outside_contents: str,
) -> None:
  root = tmp_path / "skills"
  root.mkdir()
  outside = tmp_path / "outside.md"
  outside.write_text(outside_contents, encoding="utf-8")
  (root / "escaped-skill.md").symlink_to(outside)

  catalog = DirectoryControlSkillCatalog(root)
  assert catalog.list_skills() == ()
  with pytest.raises(ControlSkillUnavailableError) as error:
    catalog.resolve_skill("escaped-skill")
  assert error.value.code == "unknown"
  assert error.value.__cause__ is None
  assert error.value.__context__ is None


def test_intra_root_symlink_alias_remains_supported(tmp_path: Path) -> None:
  root = tmp_path / "skills"
  backing = _write_skill(
    root,
    "Backing_Target",
    frontmatter="name: alias-skill\ncatalog: true",
    body="Alias body.",
  )
  (root / "alias-skill.md").symlink_to(backing)

  catalog = DirectoryControlSkillCatalog(root)
  assert tuple(summary.name for summary in catalog.list_skills()) == (
    "alias-skill",
  )
  assert catalog.resolve_skill("alias-skill").body == "Alias body."


@pytest.mark.parametrize("operation", ["list", "detail"])
def test_one_source_snapshot_stabilizes_true_to_false_mutation(
  tmp_path: Path,
  monkeypatch: pytest.MonkeyPatch,
  operation: str,
) -> None:
  root = tmp_path / "skills"
  _write_skill(
    root,
    "changing-skill",
    frontmatter="name: changing-skill\ncatalog: true",
    body="Snapshot body.",
  )
  loads: list[str] = []

  class MutatingLoader:
    def __init__(self, skills_dir: str | Path) -> None:
      self._root = Path(skills_dir)

    def load_source(self, source: str, *, path: Path):
      loads.append(path.stem)
      path.write_text(
        "---\nname: changing-skill\ncatalog: false\n---\nChanged",
        encoding="utf-8",
      )
      return SkillLoader(self._root).load_source(source, path=path)

  monkeypatch.setattr(
    "agent_gateway.directory_control_skill_catalog.SkillLoader",
    MutatingLoader,
  )
  catalog = DirectoryControlSkillCatalog(root)

  if operation == "list":
    assert tuple(summary.name for summary in catalog.list_skills()) == (
      "changing-skill",
    )
    assert catalog.list_skills() == ()
  else:
    assert catalog.resolve_skill("changing-skill").body == "Snapshot body."
    with pytest.raises(ControlSkillUnavailableError) as error:
      catalog.resolve_skill("changing-skill")
    assert error.value.code == "unknown"
    assert error.value.__cause__ is None
    assert error.value.__context__ is None
  assert loads == ["changing-skill"]


@pytest.mark.parametrize(
  "newline",
  [b"\r\n", b"\r"],
  ids=("crlf", "bare-cr"),
)
def test_visible_universal_newline_snapshot_matches_file_parser(
  tmp_path: Path,
  newline: bytes,
) -> None:
  root = tmp_path / "skills"
  root.mkdir()
  path = root / "newline-skill.md"
  path.write_bytes(
    newline.join((
      b"---",
      b"name: newline-skill",
      b"catalog: true",
      b"---",
      b"Line one.",
      b"Line two.",
      b"Line three.",
    ))
  )

  catalog = DirectoryControlSkillCatalog(root)
  assert tuple(summary.name for summary in catalog.list_skills()) == (
    "newline-skill",
  )
  detail = catalog.resolve_skill("newline-skill")
  assert detail.body == "Line one.\nLine two.\nLine three."
  assert detail.body == SkillLoader(root).load("newline-skill").system_prompt


@pytest.mark.parametrize("operation", ["list", "detail"])
def test_pinned_source_refuses_swap_to_outside_before_any_byte_read(
  tmp_path: Path,
  monkeypatch: pytest.MonkeyPatch,
  operation: str,
) -> None:
  root = tmp_path / "skills"
  target = _write_skill(
    root,
    "swapped-skill",
    frontmatter="name: swapped-skill\ncatalog: true",
    body="Inside body.",
  )
  outside = tmp_path / "outside.md"
  outside.write_text(
    "---\nname: swapped-skill\ncatalog: true\n"
    "description: OUTSIDE\n---\nOUTSIDE BODY",
    encoding="utf-8",
  )
  real_open = Path.open
  read_attempts: list[str] = []
  swapped = False

  class TrackingHandle:
    def __init__(self, handle: BinaryIO) -> None:
      self._handle = handle

    def __enter__(self) -> TrackingHandle:
      self._handle.__enter__()
      return self

    def __exit__(
      self,
      exc_type: type[BaseException] | None,
      exc_value: BaseException | None,
      traceback: TracebackType | None,
    ) -> None:
      self._handle.__exit__(exc_type, exc_value, traceback)

    def fileno(self) -> int:
      return self._handle.fileno()

    def read(self, size: int = -1) -> bytes:
      read_attempts.append("read")
      return self._handle.read(size)

    def readline(self, size: int = -1) -> bytes:
      read_attempts.append("readline")
      return self._handle.readline(size)

    def __iter__(self) -> Iterator[bytes]:
      read_attempts.append("iter")
      return iter(self._handle)

  def swapping_open(
    path: Path,
    *args: Literal["rb"],
    buffering: int = -1,
  ) -> TrackingHandle | BinaryIO:
    nonlocal swapped
    if path == target and not swapped:
      swapped = True
      target.unlink()
      target.symlink_to(outside)
      return TrackingHandle(real_open(path, "rb", buffering=buffering))
    return real_open(path, "rb", buffering=buffering)

  monkeypatch.setattr(Path, "open", swapping_open)
  catalog = DirectoryControlSkillCatalog(root)

  if operation == "list":
    assert catalog.list_skills() == ()
  else:
    with pytest.raises(ControlSkillUnavailableError) as error:
      catalog.resolve_skill("swapped-skill")
    assert error.value.code == "invalid"
    assert "OUTSIDE" not in str(error.value)
  assert swapped is True
  assert read_attempts == []


@pytest.mark.parametrize("operation", ["list", "detail"])
def test_pinned_source_refuses_configured_root_swap_before_open(
  tmp_path: Path,
  monkeypatch: pytest.MonkeyPatch,
  operation: str,
) -> None:
  root = tmp_path / "skills"
  target = _write_skill(
    root,
    "root-swap",
    frontmatter="name: root-swap\ncatalog: true",
    body="Inside body.",
  )
  outside_root = tmp_path / "outside-root"
  _write_skill(
    outside_root,
    "root-swap",
    frontmatter=(
      "name: root-swap\ncatalog: true\ndescription: OUTSIDE DESCRIPTION"
    ),
    body="OUTSIDE BODY",
  )
  displaced_root = tmp_path / "displaced-root"
  real_lstat = Path.lstat
  real_open = Path.open
  swapped = False
  opened: list[Path] = []

  def swapping_lstat(path: Path):
    nonlocal swapped
    result = real_lstat(path)
    if path == target and not swapped:
      swapped = True
      root.rename(displaced_root)
      root.symlink_to(outside_root, target_is_directory=True)
    return result

  def tracking_open(
    path: Path,
    *args: Literal["rb"],
    buffering: int = -1,
  ) -> BinaryIO:
    opened.append(path)
    return real_open(path, "rb", buffering=buffering)

  monkeypatch.setattr(Path, "lstat", swapping_lstat)
  monkeypatch.setattr(Path, "open", tracking_open)
  catalog = DirectoryControlSkillCatalog(root)

  if operation == "list":
    assert catalog.list_skills() == ()
  else:
    with pytest.raises(ControlSkillUnavailableError) as error:
      catalog.resolve_skill("root-swap")
    assert error.value.code == "invalid"
    assert "OUTSIDE" not in str(error.value)
  assert swapped is True
  assert opened == []


def test_list_does_not_resolve_blocks_but_detail_does(tmp_path: Path) -> None:
  root = tmp_path / "skills"
  _write_skill(
    root,
    "missing-block",
    frontmatter="name: missing-block",
    body="Before {{DOES_NOT_EXIST}} after.",
  )
  catalog = DirectoryControlSkillCatalog(root)

  assert tuple(summary.name for summary in catalog.list_skills()) == (
    "missing-block",
  )
  with pytest.raises(ControlSkillUnavailableError) as error:
    catalog.resolve_skill("missing-block")
  assert error.value.code == "invalid"
  assert isinstance(error.value.__cause__, FileNotFoundError)


@pytest.mark.parametrize(
  "frontmatter",
  [
    "catalog: maybe",
    "max_turns: not-an-int",
    "semantic_metadata: not-a-mapping",
    "semantic_metadata:\n  output_contracts: [not-a-mapping]",
    "metadata:\n  required_context: bad: mapping",
  ],
)
def test_broken_member_is_skipped_loudly_and_read_reports_invalid(
  tmp_path: Path,
  frontmatter: str,
  caplog: pytest.LogCaptureFixture,
) -> None:
  root = tmp_path / "skills"
  _write_skill(root, "valid-skill", frontmatter="name: valid-skill")
  _write_skill(root, "broken-skill", frontmatter=frontmatter)
  catalog = DirectoryControlSkillCatalog(root)

  with caplog.at_level(
    "WARNING",
    logger="agent_gateway.directory_control_skill_catalog",
  ):
    listing = catalog.list_skills()
  assert tuple(summary.name for summary in listing) == ("valid-skill",)
  assert any(
    "broken-skill" in record.getMessage()
    for record in caplog.records
    if record.levelname == "WARNING"
  )

  with pytest.raises(ControlSkillUnavailableError) as error:
    catalog.resolve_skill("broken-skill")
  assert error.value.code == "invalid"
  assert error.value.selector == "broken-skill"
  assert error.value.__cause__ is not None
  assert "broken-skill" not in str(error.value)


def test_raw_yaml_and_delimiter_failures_are_typed_invalid_on_read(
  tmp_path: Path,
) -> None:
  root = tmp_path / "skills"
  root.mkdir()
  catalog = DirectoryControlSkillCatalog(root)
  (root / "bad-yaml.md").write_text(
    "---\ninvalid: [\n---\nBody",
    encoding="utf-8",
  )
  assert catalog.list_skills() == ()
  with pytest.raises(ControlSkillUnavailableError) as yaml_error:
    catalog.resolve_skill("bad-yaml")
  assert yaml_error.value.code == "invalid"

  (root / "bad-yaml.md").unlink()
  (root / "bad-delimiter.md").write_text(
    "---\nname: bad-delimiter\nBody",
    encoding="utf-8",
  )
  assert catalog.list_skills() == ()
  with pytest.raises(ControlSkillUnavailableError) as delimiter_error:
    catalog.resolve_skill("bad-delimiter")
  assert delimiter_error.value.code == "invalid"


@pytest.mark.parametrize("selector", [None, 1, True, "", " padded ", "Bad_Name"])
def test_detail_rejects_invalid_selector_without_disclosure(
  tmp_path: Path,
  selector: object,
) -> None:
  catalog = DirectoryControlSkillCatalog(tmp_path / "missing")
  with pytest.raises(ControlSkillUnavailableError) as error:
    catalog.resolve_skill(selector)
  assert error.value.code == "invalid_selector"
  assert error.value.selector is selector
  assert error.value.__cause__ is None
  assert error.value.__context__ is None
  assert str(error.value) == "control skill is unavailable"


def test_directory_catalog_is_lazy_live_and_name_sorted(tmp_path: Path) -> None:
  root = tmp_path / "not-created"
  catalog = DirectoryControlSkillCatalog(root)
  assert catalog._root == root
  assert catalog.list_skills() == ()

  alpha = _write_skill(
    root,
    "z-file",
    frontmatter="name: alpha-skill\ndescription: First",
  )
  _write_skill(
    root,
    "a-file",
    frontmatter="name: zeta-skill\ndescription: Other",
  )
  assert tuple(summary.name for summary in catalog.list_skills()) == (
    "alpha-skill",
    "zeta-skill",
  )

  alpha.write_text(
    "---\nname: alpha-skill\ndescription: Changed\n---\nChanged body",
    encoding="utf-8",
  )
  assert catalog.list_skills()[0].description == "Changed"
  assert catalog.resolve_skill("z-file").body == "Changed body"


def test_profile_name_must_match_control_grammar_but_not_filename(
  tmp_path: Path,
) -> None:
  root = tmp_path / "skills"
  _write_skill(root, "file-alias", frontmatter="name: other-valid-name")
  summary = DirectoryControlSkillCatalog(root).list_skills()[0]
  assert summary.name == "other-valid-name"
  assert summary.path.endswith("/file-alias.md")

  _write_skill(root, "bad-profile", frontmatter="name: Bad_Name")
  assert tuple(
    summary.name
    for summary in DirectoryControlSkillCatalog(root).list_skills()
  ) == ("other-valid-name",)
  with pytest.raises(ControlSkillUnavailableError) as error:
    DirectoryControlSkillCatalog(root).resolve_skill("bad-profile")
  assert error.value.code == "invalid"
  assert error.value.selector == "bad-profile"


def test_list_skips_non_files_non_markdown_and_invalid_stems(tmp_path: Path) -> None:
  root = tmp_path / "skills"
  _write_skill(root, "valid-skill", frontmatter="name: valid-skill")
  _write_skill(root, "Bad_Name", frontmatter="name: ignored-skill")
  (root / "notes.txt").write_text("ignored", encoding="utf-8")
  (root / "directory.md").mkdir()

  assert tuple(
    summary.name
    for summary in DirectoryControlSkillCatalog(root).list_skills()
  ) == ("valid-skill",)


def test_directory_adapter_import_boundary_and_no_reexport() -> None:
  source_path = ROOT / "agent_gateway" / "directory_control_skill_catalog.py"
  source = source_path.read_text(encoding="utf-8")
  _assert_directory_import_boundary(source)
  tree = ast.parse(source)
  forbidden = {
    "CompiledSkillControlCatalog",
    "CompiledSkillDefinitionCatalog",
    "SkillMetadataV2",
    "get_skills_root",
  }
  assert not forbidden.intersection(
    node.id for node in ast.walk(tree) if isinstance(node, ast.Name)
  )
  package_init = (ROOT / "agent_gateway" / "__init__.py").read_text(
    encoding="utf-8"
  )
  assert "directory_control_skill_catalog" not in package_init


@pytest.mark.parametrize(
  "statement",
  [
    "import pathlib",
    "import yaml as parser",
    "from .skills import SkillLoader as Loader",
    "from .reader import SkillLoader",
    "from agent.skills.catalog import SkillDefinition",
  ],
)
def test_directory_import_guard_rejects_foreign_or_aliased_imports(
  statement: str,
) -> None:
  with pytest.raises(AssertionError):
    _assert_directory_import_boundary(statement)


def test_directory_adapter_imports_without_application_modules() -> None:
  environment = dict(os.environ)
  environment["PYTHONPATH"] = str(ROOT)
  result = subprocess.run(
    [
      sys.executable,
      "-c",
      (
        "import json, sys; "
        "from agent_gateway.directory_control_skill_catalog import "
        "DirectoryControlSkillCatalog; "
        "print(json.dumps({'module': DirectoryControlSkillCatalog.__module__, "
        "'application_modules': sorted(name for name in sys.modules "
        "if name == 'agent' or name.startswith('agent.skills') "
        "or name.startswith('api.agent'))}))"
      ),
    ],
    cwd=ROOT,
    env=environment,
    check=True,
    capture_output=True,
    text=True,
  )
  assert json.loads(result.stdout) == {
    "module": "agent_gateway.directory_control_skill_catalog",
    "application_modules": [],
  }
