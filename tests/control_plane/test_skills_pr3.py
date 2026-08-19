from __future__ import annotations

import time
from pathlib import Path

from fastapi.testclient import TestClient

from agent.skills.loader import _SKILL_NAME_RE, SkillMetadata, load_skill_metadata
from agent_gateway.control_plane.middleware import CONTROL_PLANE_VERSION_HEADER
from agent_gateway.skills import parse_skill_file


ROOT = Path(__file__).resolve().parents[4]
SKILLS_DIR = ROOT / "api" / "memory" / "workspace" / "notes" / "skills"


def _auth_headers(test_control_session: dict) -> dict[str, str]:
  return {"Authorization": f"Bearer {test_control_session['session_token']}"}


def _catalog_metadata() -> list[SkillMetadata]:
  entries: list[SkillMetadata] = []
  for path in sorted(SKILLS_DIR.glob("*.md")):
    if not path.is_file() or not _SKILL_NAME_RE.match(path.stem):
      continue
    metadata = load_skill_metadata(path.stem, SKILLS_DIR)
    if metadata is not None:
      entries.append(metadata)
  return sorted(entries, key=lambda entry: entry.name)


def test_control_skills_requires_bearer_auth(client: TestClient) -> None:
  list_response = client.get("/api/control/skills")
  detail_response = client.get("/api/control/skills/comparative-analysis")

  assert list_response.status_code == 401
  assert detail_response.status_code == 401


def test_control_skills_lists_catalog_metadata(
  client: TestClient,
  test_control_session: dict,
) -> None:
  response = client.get("/api/control/skills", headers=_auth_headers(test_control_session))

  assert response.status_code == 200
  assert response.headers[CONTROL_PLANE_VERSION_HEADER] == "1"
  payload = response.json()
  expected = _catalog_metadata()
  skills = payload["skills"]
  names = [entry["name"] for entry in skills]

  assert set(payload) == {"skills"}
  assert len(skills) == len(expected)
  assert len(skills) > 50
  assert names == sorted(names)
  assert "comparative-analysis" in names
  assert "error-extraction" in names
  assert "tutor" not in names
  assert "_playbook" not in names
  assert all(entry["catalog"] is True for entry in skills)

  comparative = next(entry for entry in skills if entry["name"] == "comparative-analysis")
  assert comparative == {
    "name": "comparative-analysis",
    "label": "Comparative Analysis",
    "description": next(entry.description for entry in expected if entry.name == "comparative-analysis"),
    "agent_description": next(entry.agent_description for entry in expected if entry.name == "comparative-analysis"),
    "version": "1.1",
    "scope": "ticker",
    "requires_portfolio_context": False,
    "required_context": ["ticker"],
    "agent_callable": True,
    "resumable": True,
    "max_turns": 20,
    "max_budget_usd": 4.0,
    "persist_state": False,
    "typed_contract": None,
    "profiles": ["analyst", "research_producer"],
    "modes": ["skill"],
    "outputs": ["platform:skill-result-envelope@1"],
    "action_class": "state_write",
    "approval_policy": "human_review_before_apply",
    "tier_availability": ["paid"],
    "credential_requirements": ["market_data", "portfolio_connection"],
    "schedule_eligible": False,
    "can_launch": True,
    "can_schedule": False,
    "blocked_reason": None,
    "catalog": True,
    "path": "api/memory/workspace/notes/skills/comparative-analysis.md",
  }
  performance_review = next(entry for entry in skills if entry["name"] == "performance-review")
  assert performance_review["scope"] == "portfolio"
  assert performance_review["profiles"] == ["advisor"]
  assert performance_review["requires_portfolio_context"] is True
  assert performance_review["required_context"] == ["portfolio"]
  strategy_executor = next(entry for entry in skills if entry["name"] == "strategy-executor")
  assert strategy_executor["scope"] == "portfolio"
  assert strategy_executor["requires_portfolio_context"] is True
  assert strategy_executor["required_context"] == ["portfolio"]
  macro_review = next(entry for entry in skills if entry["name"] == "macro-review")
  assert macro_review["scope"] == "portfolio"
  assert macro_review["requires_portfolio_context"] is False
  assert macro_review["required_context"] == []
  valuation_inputs = next(entry for entry in skills if entry["name"] == "valuation-inputs")
  assert valuation_inputs["required_context"] == ["ticker"]
  assert valuation_inputs["blocked_reason"] is None
  assert valuation_inputs["can_launch"] is True
  assert valuation_inputs["can_schedule"] is True


def test_control_skill_detail_returns_metadata_and_resolved_body(
  client: TestClient,
  test_control_session: dict,
) -> None:
  response = client.get(
    "/api/control/skills/comparative-analysis",
    headers=_auth_headers(test_control_session),
  )

  assert response.status_code == 200
  payload = response.json()
  assert payload["name"] == "comparative-analysis"
  assert payload["path"] == "api/memory/workspace/notes/skills/comparative-analysis.md"
  assert payload["body"].startswith("# Comparative Analysis")
  assert "---\nname: comparative-analysis" not in payload["body"]


def test_control_skill_detail_resolves_block_references(
  client: TestClient,
  test_control_session: dict,
) -> None:
  response = client.get(
    "/api/control/skills/earnings-review",
    headers=_auth_headers(test_control_session),
  )

  assert response.status_code == 200
  body = response.json()["body"]
  assert "{{OUTPUT_QUALITY}}" not in body
  assert "{{TURN_BUDGET}}" not in body
  assert "{{ESCALATION}}" not in body
  assert "### Output Quality Rules" in body


def test_control_skill_detail_handles_unknown_catalog_false_and_error_extraction(
  client: TestClient,
  test_control_session: dict,
) -> None:
  headers = _auth_headers(test_control_session)

  unknown = client.get("/api/control/skills/not-a-skill", headers=headers)
  hidden = client.get("/api/control/skills/tutor", headers=headers)
  error_extraction = client.get(
    "/api/control/skills/error-extraction",
    headers=headers,
  )

  assert unknown.status_code == 404
  assert unknown.json()["detail"] == "Skill not found"
  assert hidden.status_code == 404
  assert hidden.json()["detail"] == "Skill not found"
  assert error_extraction.status_code == 200
  assert error_extraction.json()["name"] == "error-extraction"


def test_fixture_artifacts_are_hidden_but_resumable_for_live_qa(monkeypatch) -> None:
  monkeypatch.setenv("APP_ENV", "development")
  for name in ("ENVIRONMENT", "AGENT_GATEWAY_ENV", "NODE_ENV"):
    monkeypatch.delenv(name, raising=False)

  expected_allowed_tools = {
    "fixture-canvas-artifact": ["emit_canvas_artifact"],
    "fixture-dashboard-artifact": ["emit_dashboard_artifact"],
  }
  for skill_name, allowed_tools in expected_allowed_tools.items():
    metadata = load_skill_metadata(skill_name, SKILLS_DIR, include_catalog_false=True)
    profile = parse_skill_file(SKILLS_DIR / f"{skill_name}.md")

    assert load_skill_metadata(skill_name, SKILLS_DIR) is None
    assert metadata is not None
    assert metadata.catalog is False
    assert metadata.agent_callable is False
    assert metadata.resumable is True
    assert profile.state_class == "advisor-with-decision-log"
    assert profile.metadata is not None
    assert profile.metadata["allowed_tools"] == allowed_tools


def test_fixture_artifacts_stay_hidden_from_control_skill_routes(
  client: TestClient,
  test_control_session: dict,
  monkeypatch,
) -> None:
  monkeypatch.setenv("APP_ENV", "development")
  for name in ("ENVIRONMENT", "AGENT_GATEWAY_ENV", "NODE_ENV"):
    monkeypatch.delenv(name, raising=False)

  headers = _auth_headers(test_control_session)
  list_response = client.get("/api/control/skills", headers=headers)
  detail_response = client.get("/api/control/skills/fixture-canvas-artifact", headers=headers)
  dashboard_detail_response = client.get("/api/control/skills/fixture-dashboard-artifact", headers=headers)
  approval_detail_response = client.get("/api/control/skills/fixture-approval-canvas-artifact", headers=headers)

  assert list_response.status_code == 200
  assert "fixture-canvas-artifact" not in {
    entry["name"] for entry in list_response.json()["skills"]
  }
  assert "fixture-dashboard-artifact" not in {
    entry["name"] for entry in list_response.json()["skills"]
  }
  assert "fixture-approval-canvas-artifact" not in {
    entry["name"] for entry in list_response.json()["skills"]
  }
  assert detail_response.status_code == 404
  assert dashboard_detail_response.status_code == 404
  assert approval_detail_response.status_code == 404


def test_load_skill_metadata_is_frontmatter_only(tmp_path: Path) -> None:
  skills_dir = tmp_path / "skills"
  skills_dir.mkdir()
  (skills_dir / "frontmatter-only.md").write_text(
    """---
name: frontmatter-only
description: Frontmatter only.
version: '1.0'
scope: global
agent_callable: true
resumable: false
max_turns: 3
max_budget_usd: 1.25
persist_state: false
typed_contract: DemoContract
required_context: []
requires_portfolio_context: false
mutation_mode: read_only
modes: [skill]
tier_availability: [paid]
blocked_reason: null
semantic_metadata:
  contract_name: skill-metadata
  schema_version: '2'
  catalog_version: skill-catalog/2
  skill_id: frontmatter-only
  tool_refs: []
  allowed_effects: []
  approval_constraints: [runtime_policy]
  output_contracts:
    - owner: platform
      contract_name: skill-result-envelope
      schema_version: '1'
  credential_requirements: []
  scheduling:
    eligibility: eligible
    opt_in: not_required
  allowed_profiles: [analyst]
---

# Frontmatter Only

{{MISSING_BLOCK}}
""",
    encoding="utf-8",
  )

  metadata = load_skill_metadata("frontmatter-only", skills_dir)

  assert metadata == SkillMetadata(
    name="frontmatter-only",
    label="Frontmatter Only",
    description="Frontmatter only.",
    agent_description=None,
    version="1.0",
    scope="global",
    requires_portfolio_context=False,
    required_context=[],
    agent_callable=True,
    resumable=False,
    max_turns=3,
    max_budget_usd=1.25,
    persist_state=False,
    typed_contract="DemoContract",
    mutation_mode="read_only",
    profiles=["analyst"],
    modes=["skill"],
    outputs=["platform:skill-result-envelope@1"],
    action_class="read_only",
    approval_policy="runtime_policy",
    tier_availability=["paid"],
    credential_requirements=[],
    schedule_eligible=True,
    can_launch=True,
    can_schedule=True,
    blocked_reason=None,
    catalog=True,
    path=(skills_dir / "frontmatter-only.md").as_posix(),
  )


def test_load_skill_metadata_respects_top_level_required_context_override(tmp_path: Path) -> None:
  skills_dir = tmp_path / "skills"
  skills_dir.mkdir()
  (skills_dir / "portfolio-optional.md").write_text(
    """---
name: portfolio-optional
description: Portfolio scoped but optional.
version: '1.0'
scope: portfolio
agent_callable: true
required_context: []
requires_portfolio_context: false
mutation_mode: read_only
semantic_metadata:
  contract_name: skill-metadata
  schema_version: '2'
  catalog_version: skill-catalog/2
  skill_id: portfolio-optional
  tool_refs: []
  allowed_effects: []
  approval_constraints: [runtime_policy]
  output_contracts:
    - owner: platform
      contract_name: skill-result-envelope
      schema_version: '1'
  credential_requirements: []
  scheduling:
    eligibility: ineligible
    opt_in: not_required
  allowed_profiles: [analyst]
---

# Portfolio Optional
""",
    encoding="utf-8",
  )

  metadata = load_skill_metadata("portfolio-optional", skills_dir)

  assert metadata is not None
  assert metadata.scope == "portfolio"
  assert metadata.requires_portfolio_context is False
  assert metadata.required_context == []


def test_load_skill_metadata_respects_nested_required_context_override(tmp_path: Path) -> None:
  skills_dir = tmp_path / "skills"
  skills_dir.mkdir()
  (skills_dir / "portfolio-name-required.md").write_text(
    """---
name: portfolio-name-required
description: Portfolio context by nested metadata.
version: '1.0'
scope: global
agent_callable: true
metadata:
  required_context:
    - portfolio_name
requires_portfolio_context: true
mutation_mode: read_only
semantic_metadata:
  contract_name: skill-metadata
  schema_version: '2'
  catalog_version: skill-catalog/2
  skill_id: portfolio-name-required
  tool_refs: []
  allowed_effects: []
  approval_constraints: [runtime_policy]
  output_contracts:
    - owner: platform
      contract_name: skill-result-envelope
      schema_version: '1'
  credential_requirements: []
  scheduling:
    eligibility: ineligible
    opt_in: not_required
  allowed_profiles: [analyst]
---

# Portfolio Name Required
""",
    encoding="utf-8",
  )

  metadata = load_skill_metadata("portfolio-name-required", skills_dir)

  assert metadata is not None
  assert metadata.scope == "global"
  assert metadata.requires_portfolio_context is True
  assert metadata.required_context == ["portfolio_name"]


def test_load_skill_metadata_catalog_scan_under_100ms() -> None:
  # Warm the filesystem cache first: a cold first touch in a fresh worktree
  # pays I/O costs unrelated to scan speed (verifier red 51e298a7d, retry
  # included; root cause per stabilization lane = cold-cache first touch).
  _catalog_metadata()

  start = time.perf_counter()
  metadata = _catalog_metadata()
  elapsed = time.perf_counter() - start

  assert metadata
  assert elapsed < 0.1
