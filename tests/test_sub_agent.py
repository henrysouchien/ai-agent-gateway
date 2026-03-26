import asyncio
import datetime
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
PKG_DIR = ROOT / "packages" / "claude-gateway"
if str(PKG_DIR) not in sys.path:
  sys.path.insert(0, str(PKG_DIR))

from claude_gateway.skills import SkillLoader
from claude_gateway.sub_agent import (
  _DEFAULT_EXCLUDED_TOOLS,
  _DEFAULT_SYSTEM_PROMPT_TEMPLATE,
  _SKILL_SYSTEM_PROMPT_TEMPLATE,
  make_run_agent_handler,
  make_run_agent_tool_def,
)


def _run(coro):
  return asyncio.run(coro)


async def _dummy_tool(_tool_input, **_kwargs):
  return {"ok": True}, None


class _StubMcpClient:
  def is_mcp_tool(self, _name: str) -> bool:
    return False

  async def call_tool(self, _name: str, _tool_input: dict):
    raise AssertionError("unexpected MCP tool dispatch")


class _StubRunner:
  def __init__(self) -> None:
    self._full_session_id = "session-sub-agent"
    self.calls: list[dict] = []

  async def spawn_sub_agent(self, task: str, **kwargs):
    self.calls.append({"task": task, **kwargs})
    return {"response": "ok"}, None


def _write_skill(skills_dir: Path, name: str, body: str) -> None:
  skills_dir.mkdir(parents=True, exist_ok=True)
  (skills_dir / f"{name}.md").write_text(body, encoding="utf-8")


def test_make_run_agent_handler_loads_skill_profile_and_filters_tools(tmp_path: Path) -> None:
  skills_dir = tmp_path / "skills"
  _write_skill(
    skills_dir,
    "deep-research",
    "---\nmodel: claude-opus-4-6\nmax_turns: 7\ntimeout: 12.5\n---\nUse multiple sources.",
  )
  runner = _StubRunner()
  keep_tool = _dummy_tool
  drop_tool = _dummy_tool
  handler = make_run_agent_handler(
    [runner],
    skill_loader=SkillLoader(skills_dir),
    mcp_client=_StubMcpClient(),
    local_tool_handlers={
      "keep_tool": keep_tool,
      "drop_tool": drop_tool,
      "run_agent": _dummy_tool,
    },
    excluded_tools={"drop_tool"},
    default_model="claude-sonnet-4-6",
    allowed_models={"claude-sonnet-4-6", "claude-opus-4-6"},
  )

  result, error = _run(handler({"agent": "deep-research", "task": "Analyze filings"}, call_index=3))

  assert error is None
  assert result == {"response": "ok"}
  assert len(runner.calls) == 1

  call = runner.calls[0]
  dispatcher = call["dispatcher"]
  assert call["task"] == "Analyze filings"
  assert call["model"] == "claude-opus-4-6"
  assert call["system_prompt"] == _SKILL_SYSTEM_PROMPT_TEMPLATE.format(
    skill_prompt="Use multiple sources.",
    date=datetime.date.today().isoformat(),
  )
  assert call["excluded_tools"] == _DEFAULT_EXCLUDED_TOOLS | {"drop_tool"}
  assert call["max_turns"] == 7
  assert call["timeout"] == 12.5
  assert call["client_timeout"] == 90
  assert call["max_tokens"] == 32000
  assert call["call_index"] == 3
  assert dispatcher._local == {"keep_tool": keep_tool}
  assert dispatcher._needs_approval("keep_tool", {}, "") is False
  assert dispatcher._session_id == runner._full_session_id


def test_make_run_agent_handler_uses_anonymous_defaults_for_blank_agent(tmp_path: Path) -> None:
  runner = _StubRunner()
  handler = make_run_agent_handler(
    [runner],
    skill_loader=SkillLoader(tmp_path / "skills"),
    mcp_client=_StubMcpClient(),
    local_tool_handlers={"keep_tool": _dummy_tool},
    default_model="claude-opus-4-6",
    default_max_turns=9,
    default_timeout=42.0,
  )

  result, error = _run(handler({"agent": "   ", "task": "Quick question"}))

  assert error is None
  assert result == {"response": "ok"}
  call = runner.calls[0]
  assert call["model"] == "claude-opus-4-6"
  assert call["system_prompt"] == _DEFAULT_SYSTEM_PROMPT_TEMPLATE.format(
    date=datetime.date.today().isoformat(),
  )
  assert call["max_turns"] == 9
  assert call["timeout"] == 42.0


def test_make_run_agent_handler_returns_not_available_for_named_agents_without_skills() -> None:
  handler = make_run_agent_handler(
    [_StubRunner()],
    skill_loader=None,
    mcp_client=_StubMcpClient(),
    local_tool_handlers={},
  )

  result, error = _run(handler({"agent": "research", "task": "Collect"}))

  assert result is None
  assert error == {"code": "not_available", "message": "Named agents not available"}


def test_make_run_agent_handler_returns_not_found_for_unknown_skill(tmp_path: Path) -> None:
  handler = make_run_agent_handler(
    [_StubRunner()],
    skill_loader=SkillLoader(tmp_path / "skills"),
    mcp_client=_StubMcpClient(),
    local_tool_handlers={},
  )

  result, error = _run(handler({"agent": "missing", "task": "Collect"}))

  assert result is None
  assert error is not None
  assert error["code"] == "not_found"
  assert "Skill 'missing' not found." in error["message"]


def test_make_run_agent_handler_validates_effective_skill_model(tmp_path: Path) -> None:
  skills_dir = tmp_path / "skills"
  _write_skill(skills_dir, "research", "---\nmodel: custom-model\n---\nResearch.")
  handler = make_run_agent_handler(
    [_StubRunner()],
    skill_loader=SkillLoader(skills_dir),
    mcp_client=_StubMcpClient(),
    local_tool_handlers={},
    allowed_models={"claude-sonnet-4-6"},
  )

  result, error = _run(handler({"agent": "research", "task": "Collect"}))

  assert result is None
  assert error == {
    "code": "invalid_input",
    "message": "Invalid model 'custom-model' for skill 'research'",
  }


def test_make_run_agent_handler_returns_internal_error_when_runner_missing() -> None:
  handler = make_run_agent_handler(
    [None],
    skill_loader=None,
    mcp_client=_StubMcpClient(),
    local_tool_handlers={},
  )

  result, error = _run(handler({"task": "Collect"}))

  assert result is None
  assert error == {"code": "internal_error", "message": "Sub-agent runner not initialized"}


def test_make_run_agent_handler_rejects_empty_task() -> None:
  handler = make_run_agent_handler(
    [_StubRunner()],
    skill_loader=None,
    mcp_client=_StubMcpClient(),
    local_tool_handlers={},
  )

  result, error = _run(handler({"task": ""}))

  assert result is None
  assert error == {"code": "invalid_input", "message": "task is required"}


def test_make_run_agent_handler_rejects_non_string_agent() -> None:
  handler = make_run_agent_handler(
    [_StubRunner()],
    skill_loader=None,
    mcp_client=_StubMcpClient(),
    local_tool_handlers={},
  )

  result, error = _run(handler({"agent": 123, "task": "Collect"}))

  assert result is None
  assert error == {"code": "invalid_input", "message": "agent must be a string"}


def test_make_run_agent_handler_rejects_non_string_model() -> None:
  handler = make_run_agent_handler(
    [_StubRunner()],
    skill_loader=None,
    mcp_client=_StubMcpClient(),
    local_tool_handlers={},
  )

  result, error = _run(handler({"task": "Collect", "model": 123}))

  assert result is None
  assert error == {"code": "invalid_input", "message": "model must be a string"}


def test_make_run_agent_handler_rejects_invalid_raw_model() -> None:
  handler = make_run_agent_handler(
    [_StubRunner()],
    skill_loader=None,
    mcp_client=_StubMcpClient(),
    local_tool_handlers={},
    allowed_models={"claude-sonnet-4-6"},
  )

  result, error = _run(handler({"task": "Collect", "model": "bad-model"}))

  assert result is None
  assert error == {"code": "invalid_input", "message": "Invalid model: bad-model"}


def test_make_run_agent_handler_returns_invalid_skill_for_malformed_yaml(tmp_path: Path) -> None:
  skills_dir = tmp_path / "skills"
  _write_skill(skills_dir, "broken", "---\nmodel: [\n---\nBroken.")
  handler = make_run_agent_handler(
    [_StubRunner()],
    skill_loader=SkillLoader(skills_dir),
    mcp_client=_StubMcpClient(),
    local_tool_handlers={},
  )

  result, error = _run(handler({"agent": "broken", "task": "Collect"}))

  assert result is None
  assert error is not None
  assert error["code"] == "invalid_skill"


def test_make_run_agent_handler_allows_any_model_when_allowed_models_empty(tmp_path: Path) -> None:
  runner = _StubRunner()
  handler = make_run_agent_handler(
    [runner],
    skill_loader=SkillLoader(tmp_path / "skills"),
    mcp_client=_StubMcpClient(),
    local_tool_handlers={},
    default_model="claude-sonnet-4-6",
    allowed_models=set(),
  )

  result, error = _run(handler({"task": "Collect", "model": "custom-model"}))

  assert error is None
  assert result == {"response": "ok"}
  assert runner.calls[0]["model"] == "custom-model"


def test_make_run_agent_tool_def_includes_skill_names(tmp_path: Path) -> None:
  skills_dir = tmp_path / "skills"
  _write_skill(skills_dir, "alpha", "Alpha.")
  _write_skill(skills_dir, "beta", "Beta.")

  tool_def = make_run_agent_tool_def(SkillLoader(skills_dir))

  assert tool_def["description"].endswith("Available agents: alpha, beta.")
  assert tool_def["input_schema"]["properties"]["agent"]["description"].endswith("One of: alpha, beta.")


def test_make_run_agent_tool_def_has_no_skill_suffix_without_skills() -> None:
  tool_def = make_run_agent_tool_def()

  assert "Available agents:" not in tool_def["description"]
  assert "One of:" not in tool_def["input_schema"]["properties"]["agent"]["description"]
