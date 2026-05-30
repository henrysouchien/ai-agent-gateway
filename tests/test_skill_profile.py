import sys
import textwrap
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
PKG_DIR = ROOT / "packages" / "agent-gateway"
API_DIR = ROOT / "api"
if str(PKG_DIR) not in sys.path:
  sys.path.insert(0, str(PKG_DIR))
if str(API_DIR) not in sys.path:
  sys.path.insert(0, str(API_DIR))

from agent_gateway.skills import (
  AGENT_DESCRIPTION_MAX_CHARS,
  AGENT_DESCRIPTION_PLACEHOLDER,
  SKILL_STATE_CLASSES,
  SkillLoader,
  SkillProfile,
  parse_skill_file,
)


def _write_skill(tmp_path: Path, frontmatter: str | None, *, body: str = "# Skill\n\nPrompt") -> Path:
  skill_path = tmp_path / "test-skill.md"
  if frontmatter is None:
    text = body
  else:
    text = f"---\n{textwrap.dedent(frontmatter).strip()}\n---\n\n{body}\n"
  skill_path.write_text(text, encoding="utf-8")
  return skill_path


def _write_named_skill(skills_dir: Path, name: str, frontmatter: str | None, *, body: str = "# Skill") -> None:
  skills_dir.mkdir(parents=True, exist_ok=True)
  if frontmatter is None:
    text = body
  else:
    text = f"---\n{textwrap.dedent(frontmatter).strip()}\n---\n\n{body}\n"
  (skills_dir / f"{name}.md").write_text(text, encoding="utf-8")


def test_parse_lifts_metadata_keys(tmp_path: Path) -> None:
  skill_path = _write_skill(
    tmp_path,
    """
    name: typed-metadata
    metadata:
      mcp_servers:
        - gmail
        - slack
      mcp_tools:
        gmail:
          - search_messages
          - search_messages
          - send_message
        slack: post_message
      session_inject_servers:
        - gmail
      timeout_overrides:
        gmail: "30"
        slack: 45
      state_dir: daily-scan
      max_budget_usd: "0.75"
      max_retries: 2
      initial_message: Run the workflow.
      delivery_label: Daily Scan
    """,
  )

  profile = parse_skill_file(skill_path)

  assert profile.mcp_servers == ["gmail", "slack"]
  assert profile.mcp_tools == {
    "gmail": ["search_messages", "send_message"],
    "slack": ["post_message"],
  }
  assert profile.session_inject_servers == ["gmail"]
  assert profile.timeout_overrides == {"gmail": 30, "slack": 45}
  assert profile.state_dir == "daily-scan"
  assert profile.max_budget_usd == 0.75
  assert profile.max_retries == 2
  assert profile.initial_message == "Run the workflow."
  assert profile.delivery_label == "Daily Scan"
  assert profile.metadata == {
    "mcp_servers": ["gmail", "slack"],
    "mcp_tools": {
      "gmail": ["search_messages", "send_message"],
      "slack": ["post_message"],
    },
    "session_inject_servers": ["gmail"],
    "timeout_overrides": {"gmail": 30, "slack": 45},
    "state_dir": "daily-scan",
    "max_budget_usd": 0.75,
    "max_retries": 2,
    "initial_message": "Run the workflow.",
    "delivery_label": "Daily Scan",
  }


def test_list_callable_skills_with_descriptions_filters_truncates_and_placeholders(
  tmp_path: Path,
  caplog: pytest.LogCaptureFixture,
) -> None:
  skills_dir = tmp_path / "skills"
  long_description = "x" * (AGENT_DESCRIPTION_MAX_CHARS + 20)
  _write_named_skill(
    skills_dir,
    "alpha",
    """
    agent_callable: true
    agent_description: Alpha reviews earnings.
    """,
  )
  _write_named_skill(
    skills_dir,
    "beta",
    """
    agent_callable: true
    """,
  )
  _write_named_skill(
    skills_dir,
    "gamma",
    f"""
    agent_callable: true
    agent_description: {long_description}
    """,
  )
  _write_named_skill(
    skills_dir,
    "not-callable",
    """
    agent_description: Hidden from callable catalog.
    """,
  )

  entries = SkillLoader(skills_dir).list_callable_skills_with_descriptions()

  assert entries[0] == ("alpha", "Alpha reviews earnings.")
  assert entries[1] == ("beta", AGENT_DESCRIPTION_PLACEHOLDER)
  assert entries[2][0] == "gamma"
  assert len(entries[2][1]) == AGENT_DESCRIPTION_MAX_CHARS
  assert entries[2][1].endswith("…")
  assert "not-callable" not in dict(entries)
  assert "missing agent_description" in caplog.text
  assert "rendered output will be truncated" in caplog.text


def test_parse_lifts_top_level_keys(tmp_path: Path) -> None:
  skill_path = _write_skill(
    tmp_path,
    """
    name: typed-top-level
    mcp_servers:
      - github
    session_inject_servers:
      - github
    timeout_overrides:
      github: "90"
    state_dir: top-level-state
    max_budget_usd: 1.25
    max_retries: 3
    initial_message: Run from top level.
    delivery_label: Top Level
    """,
  )

  profile = parse_skill_file(skill_path)

  assert profile.mcp_servers == ["github"]
  assert profile.session_inject_servers == ["github"]
  assert profile.timeout_overrides == {"github": 90}
  assert profile.state_dir == "top-level-state"
  assert profile.max_budget_usd == 1.25
  assert profile.max_retries == 3
  assert profile.initial_message == "Run from top level."
  assert profile.delivery_label == "Top Level"
  assert profile.metadata == {
    "mcp_servers": ["github"],
    "session_inject_servers": ["github"],
    "timeout_overrides": {"github": 90},
    "state_dir": "top-level-state",
    "max_budget_usd": 1.25,
    "max_retries": 3,
    "initial_message": "Run from top level.",
    "delivery_label": "Top Level",
  }


def test_parse_agent_fields_from_frontmatter(tmp_path: Path) -> None:
  skill_path = _write_skill(
    tmp_path,
    """
    name: agent-fields
    agent_callable: true
    agent_description: Agent catalog description
    mode: recommend
    extra_excluded_tools:
      - execute_trade
      - preview_trade
    tool_packs_enabled: false
    """,
  )

  profile = parse_skill_file(skill_path)

  assert profile.agent_callable is True
  assert profile.agent_description == "Agent catalog description"
  assert profile.mode == "recommend"
  assert profile.extra_excluded_tools == {"execute_trade", "preview_trade"}
  assert profile.tool_packs_enabled is False


def test_parse_state_class_from_frontmatter(tmp_path: Path) -> None:
  skill_path = _write_skill(
    tmp_path,
    """
    name: stateful-skill
    state_class: producer
    """,
  )

  profile = parse_skill_file(skill_path)

  assert profile.state_class == "producer"
  assert "producer" in SKILL_STATE_CLASSES


def test_parse_invalid_state_class_raises(tmp_path: Path) -> None:
  skill_path = _write_skill(
    tmp_path,
    """
    state_class: writes-sometimes
    """,
  )

  with pytest.raises(ValueError, match="state_class"):
    parse_skill_file(skill_path)


def test_parse_resumable_defaults_load_cleanly(tmp_path: Path) -> None:
  skill_path = _write_skill(tmp_path, None)

  profile = parse_skill_file(skill_path)

  assert profile.resumable is False
  assert profile.resume_mcp_session_reset_ok is False
  assert profile.state_class is None


def test_parse_resumable_with_session_injection_requires_reset_ack(tmp_path: Path) -> None:
  skill_path = _write_skill(
    tmp_path,
    """
    resumable: true
    session_inject_servers:
      - browser
    """,
  )

  with pytest.raises(ValueError, match="resume_mcp_session_reset_ok"):
    parse_skill_file(skill_path)


def test_parse_resumable_with_session_injection_and_reset_ack_loads(tmp_path: Path) -> None:
  skill_path = _write_skill(
    tmp_path,
    """
    resumable: true
    resume_mcp_session_reset_ok: true
    session_inject_servers:
      - browser
    """,
  )

  profile = parse_skill_file(skill_path)

  assert profile.resumable is True
  assert profile.resume_mcp_session_reset_ok is True
  assert profile.session_inject_servers == ["browser"]


def test_mixed_known_and_unknown_metadata(tmp_path: Path) -> None:
  skill_path = _write_skill(
    tmp_path,
    """
    name: mixed-metadata
    metadata:
      mcp_servers:
        - gmail
      state_dir: mixed-state
      owner: analyst
    custom_limit: 5
    """,
  )

  profile = parse_skill_file(skill_path)

  assert profile.mcp_servers == ["gmail"]
  assert profile.state_dir == "mixed-state"
  assert profile.metadata == {
    "owner": "analyst",
    "custom_limit": 5,
    "mcp_servers": ["gmail"],
    "state_dir": "mixed-state",
  }


def test_timeout_overrides_coercion(tmp_path: Path) -> None:
  skill_path = _write_skill(
    tmp_path,
    """
    metadata:
      timeout_overrides:
        gmail: "120"
        github: 45.0
        "": 30
    """,
  )

  profile = parse_skill_file(skill_path)

  assert profile.timeout_overrides == {"gmail": 120, "github": 45}
  assert profile.metadata == {"timeout_overrides": {"gmail": 120, "github": 45}}


def test_timeout_overrides_invalid(tmp_path: Path) -> None:
  skill_path = _write_skill(
    tmp_path,
    """
    metadata:
      timeout_overrides:
        - gmail
        - 30
    """,
  )

  with pytest.raises(ValueError, match="timeout_overrides"):
    parse_skill_file(skill_path)


def test_max_budget_coercion(tmp_path: Path) -> None:
  skill_path = _write_skill(
    tmp_path,
    """
    metadata:
      max_budget_usd: "0.75"
    """,
  )

  profile = parse_skill_file(skill_path)

  assert profile.max_budget_usd == 0.75
  assert profile.metadata == {"max_budget_usd": 0.75}


def test_max_retries_coercion(tmp_path: Path) -> None:
  skill_path = _write_skill(
    tmp_path,
    """
    metadata:
      max_retries: 3.0
    """,
  )

  profile = parse_skill_file(skill_path)

  assert profile.max_retries == 3
  assert profile.metadata == {"max_retries": 3}


def test_none_defaults(tmp_path: Path) -> None:
  skill_path = _write_skill(
    tmp_path,
    """
    name: defaults
    """,
  )

  profile = parse_skill_file(skill_path)

  assert profile.metadata is None
  assert profile.mcp_servers is None
  assert profile.session_inject_servers is None
  assert profile.timeout_overrides is None
  assert profile.state_dir is None
  assert profile.max_budget_usd is None
  assert profile.max_retries is None
  assert profile.initial_message is None
  assert profile.delivery_label is None
  assert profile.agent_callable is False
  assert profile.agent_description is None
  assert profile.mode == "full"
  assert profile.extra_excluded_tools == set()
  assert profile.tool_packs_enabled is True
  assert profile.state_class is None


def test_dataclass_construction_defaults() -> None:
  profile = SkillProfile(name="x", system_prompt="y")

  assert profile.metadata is None
  assert profile.mcp_servers is None
  assert profile.session_inject_servers is None
  assert profile.timeout_overrides is None
  assert profile.state_dir is None
  assert profile.max_budget_usd is None
  assert profile.max_retries is None
  assert profile.initial_message is None
  assert profile.delivery_label is None
  assert profile.agent_callable is False
  assert profile.agent_description is None
  assert profile.mode == "full"
  assert profile.extra_excluded_tools == set()
  assert profile.tool_packs_enabled is True
  assert profile.state_class is None


def test_positional_construction_compat() -> None:
  profile = SkillProfile("name", "prompt", None, None, None, None, None, False, None, False, {"custom": 1})

  assert profile.name == "name"
  assert profile.system_prompt == "prompt"
  assert profile.metadata == {"custom": 1}
  assert profile.mcp_servers is None
  assert profile.delivery_label is None
  assert profile.agent_callable is False
  assert profile.mode == "full"
  assert profile.extra_excluded_tools == set()
  assert profile.tool_packs_enabled is True


def test_agent_profile_subclass_compat() -> None:
  profile = SkillProfile(name="agent", system_prompt="", agent_description="Agent description")

  assert profile.name == "agent"
  assert profile.agent_description == "Agent description"
  assert profile.metadata is None
  assert profile.mcp_servers is None
  assert profile.session_inject_servers is None
  assert profile.timeout_overrides is None
  assert profile.state_dir is None
  assert profile.max_budget_usd is None
  assert profile.max_retries is None
  assert profile.initial_message is None
  assert profile.delivery_label is None


def test_parse_invalid_mode_raises(tmp_path: Path) -> None:
  skill_path = _write_skill(
    tmp_path,
    """
    mode: bogus
    """,
  )

  with pytest.raises(ValueError, match="mode"):
    parse_skill_file(skill_path)


def test_metadata_retains_known_keys(tmp_path: Path) -> None:
  skill_path = _write_skill(
    tmp_path,
    """
    metadata:
      mcp_servers:
        - gmail
      delivery_label:
    """,
  )

  profile = parse_skill_file(skill_path)

  assert profile.metadata is not None
  assert profile.metadata["mcp_servers"] == ["gmail"]
  assert "delivery_label" in profile.metadata
  assert profile.metadata["delivery_label"] is None


def test_empty_mcp_servers_is_none(tmp_path: Path) -> None:
  skill_path = _write_skill(
    tmp_path,
    """
    mcp_servers: []
    """,
  )

  profile = parse_skill_file(skill_path)

  assert profile.mcp_servers is None
  assert profile.metadata == {"mcp_servers": None}


def test_empty_mcp_tools_is_none(tmp_path: Path) -> None:
  skill_path = _write_skill(
    tmp_path,
    """
    mcp_tools:
      portfolio-mcp: []
    """,
  )

  profile = parse_skill_file(skill_path)

  assert profile.mcp_tools is None
  assert profile.metadata == {"mcp_tools": None}


def test_invalid_mcp_tools_shape_rejected(tmp_path: Path) -> None:
  skill_path = _write_skill(
    tmp_path,
    """
    mcp_tools:
      - get_model_insights
    """,
  )

  with pytest.raises(ValueError, match="mcp_tools"):
    parse_skill_file(skill_path)
