from __future__ import annotations

import time
from pathlib import Path

from fastapi.testclient import TestClient

from agent.skills.loader import _SKILL_NAME_RE, SkillMetadata, load_skill_metadata
from agent_gateway.control_plane.middleware import CONTROL_PLANE_VERSION_HEADER


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
  assert "tutor" not in names
  assert "_playbook" not in names
  assert all(entry["catalog"] is True for entry in skills)

  comparative = next(entry for entry in skills if entry["name"] == "comparative-analysis")
  assert comparative == {
    "name": "comparative-analysis",
    "description": next(entry.description for entry in expected if entry.name == "comparative-analysis"),
    "agent_description": next(entry.agent_description for entry in expected if entry.name == "comparative-analysis"),
    "version": "1.0",
    "scope": "ticker",
    "agent_callable": True,
    "resumable": True,
    "max_turns": 20,
    "max_budget_usd": 4.0,
    "persist_state": True,
    "typed_contract": None,
    "catalog": True,
    "path": "api/memory/workspace/notes/skills/comparative-analysis.md",
  }


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


def test_control_skill_detail_404s_for_unknown_and_catalog_false(
  client: TestClient,
  test_control_session: dict,
) -> None:
  headers = _auth_headers(test_control_session)

  unknown = client.get("/api/control/skills/not-a-skill", headers=headers)
  hidden = client.get("/api/control/skills/tutor", headers=headers)

  assert unknown.status_code == 404
  assert unknown.json()["detail"] == "Skill not found"
  assert hidden.status_code == 404
  assert hidden.json()["detail"] == "Skill not found"


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
---

# Frontmatter Only

{{MISSING_BLOCK}}
""",
    encoding="utf-8",
  )

  metadata = load_skill_metadata("frontmatter-only", skills_dir)

  assert metadata == SkillMetadata(
    name="frontmatter-only",
    description="Frontmatter only.",
    agent_description=None,
    version="1.0",
    scope="global",
    agent_callable=True,
    resumable=False,
    max_turns=3,
    max_budget_usd=1.25,
    persist_state=False,
    typed_contract="DemoContract",
    catalog=True,
    path=(skills_dir / "frontmatter-only.md").as_posix(),
  )


def test_load_skill_metadata_catalog_scan_under_100ms() -> None:
  start = time.perf_counter()
  metadata = _catalog_metadata()
  elapsed = time.perf_counter() - start

  assert metadata
  assert elapsed < 0.1
