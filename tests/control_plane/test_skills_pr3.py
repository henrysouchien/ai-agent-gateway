from __future__ import annotations

import time
from functools import cache
from pathlib import Path, PurePosixPath

from fastapi.testclient import TestClient

from agent.skills.composition import compile_skill_application
from agent.shared.tool_registration import (
  build_product_tool_registration_composition,
)
from agent_gateway.control_skill_catalog import ControlSkillSummary
from agent_gateway.control_plane.middleware import CONTROL_PLANE_VERSION_HEADER


ROOT = Path(__file__).resolve().parents[4]
SKILLS_DIR = ROOT / "api" / "memory" / "workspace" / "notes" / "skills"


def _auth_headers(test_control_session: dict) -> dict[str, str]:
  return {"Authorization": f"Bearer {test_control_session['session_token']}"}


@cache
def _catalog_metadata() -> list[ControlSkillSummary]:
  tool_registration = build_product_tool_registration_composition()
  application = compile_skill_application(
    skills_root=SKILLS_DIR,
    source_prefix=PurePosixPath(
      "api/memory/workspace/notes/skills"
    ),
    tool_registration_catalog=tool_registration.catalog,
  )
  return list(application.control_catalog.list_skills())


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
    "profiles": ["analyst", "community", "research_producer"],
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


def test_product_control_catalog_excludes_test_only_fixtures() -> None:
  names = {entry.name for entry in _catalog_metadata()}
  assert "fixture-canvas-artifact" not in names
  assert "fixture-dashboard-artifact" not in names
  assert "fixture-approval-canvas-artifact" not in names


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


def test_compiled_control_catalog_scan_under_100ms() -> None:
  # Warm the filesystem cache first: a cold first touch in a fresh worktree
  # pays I/O costs unrelated to scan speed (verifier red 51e298a7d, retry
  # included; root cause per stabilization lane = cold-cache first touch).
  _catalog_metadata()

  start = time.perf_counter()
  metadata = _catalog_metadata()
  elapsed = time.perf_counter() - start

  assert metadata
  assert elapsed < 0.1
