import asyncio
import datetime
import sys
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[3]
PKG_DIR = ROOT / "packages" / "agent-gateway"
if str(PKG_DIR) not in sys.path:
  sys.path.insert(0, str(PKG_DIR))

from agent_gateway.skills import SkillLoader
from agent_gateway import EventLog
from agent_gateway.session import GatewaySession
from agent_gateway.sub_agent import (
  _DEFAULT_EXCLUDED_TOOLS,
  _DEFAULT_SYSTEM_PROMPT_TEMPLATE,
  _SKILL_SYSTEM_PROMPT_TEMPLATE,
  make_get_background_result_handler,
  make_get_background_result_tool_def,
  make_run_agent_handler,
  make_run_agent_tool_def,
)
from agent_gateway.tool_dispatcher import ToolExecutionContext


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
    self.background_calls: list[dict] = []

  async def spawn_sub_agent(self, task: str, **kwargs):
    self.calls.append({"task": task, **kwargs})
    return {"response": "ok"}, None

  async def _register_background_task(self, **kwargs):
    self.background_calls.append(dict(kwargs))
    return {"task_id": "bg_0", "status": "running"}, None

  async def get_background_result(self, tool_input: dict[str, Any]):
    return {"task_id": tool_input["task_id"], "status": "completed"}, None


def _write_skill(skills_dir: Path, name: str, body: str) -> None:
  skills_dir.mkdir(parents=True, exist_ok=True)
  (skills_dir / f"{name}.md").write_text(body, encoding="utf-8")


def _callable_skill(frontmatter: str = "", body: str = "Use multiple sources.") -> str:
  frontmatter = frontmatter.strip()
  lines = ["---", "agent_callable: true", "agent_description: Callable test skill."]
  if frontmatter:
    lines.extend(frontmatter.splitlines())
  lines.extend(["---", body])
  return "\n".join(lines)


def test_make_run_agent_handler_loads_skill_profile_and_filters_tools(tmp_path: Path) -> None:
  skills_dir = tmp_path / "skills"
  _write_skill(
    skills_dir,
    "deep-research",
    _callable_skill("model: claude-opus-4-6\nmax_turns: 7\ntimeout: 12.5"),
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


def test_make_run_agent_handler_default_timeout_is_none() -> None:
  runner = _StubRunner()
  handler = make_run_agent_handler(
    [runner],
    skill_loader=None,
    mcp_client=_StubMcpClient(),
    local_tool_handlers={},
  )

  result, error = _run(handler({"task": "Quick question"}))

  assert error is None
  assert result == {"response": "ok"}
  assert runner.calls[0]["timeout"] is None


def test_make_run_agent_handler_forwards_needs_approval_to_child_dispatcher() -> None:
  runner = _StubRunner()
  handler = make_run_agent_handler(
    [runner],
    skill_loader=None,
    mcp_client=_StubMcpClient(),
    local_tool_handlers={"keep_tool": _dummy_tool},
    needs_approval=lambda name, _tool_input, _qualifier: name == "sensitive_tool",
  )

  result, error = _run(handler({"task": "Quick question"}))

  assert error is None
  assert result == {"response": "ok"}
  dispatcher = runner.calls[0]["dispatcher"]
  assert dispatcher._needs_approval("sensitive_tool", {}, "") is True
  assert dispatcher._needs_approval("keep_tool", {}, "") is False


def test_make_run_agent_handler_copies_user_id_and_auth_config_to_sub_session() -> None:
  runner = _StubRunner()
  parent_auth_config = {"provider": "anthropic", "billing_mode": "byok", "api_key": "key"}
  parent_session = GatewaySession(
    session_id="sess_parent",
    api_key_hash="hash",
    created_at=1,
    expires_at=2,
    user_id="alice",
    user_email="alice@example.com",
    auth_config=parent_auth_config,
  )
  handler = make_run_agent_handler(
    [runner],
    parent_session=parent_session,
    skill_loader=None,
    mcp_client=_StubMcpClient(),
    local_tool_handlers={},
  )

  result, error = _run(handler({"task": "Collect"}))

  assert error is None
  assert result == {"response": "ok"}
  sub_session = runner.calls[0]["sub_session"]
  assert sub_session.user_id == "alice"
  assert sub_session.user_email == "alice@example.com"
  assert sub_session.auth_config is parent_auth_config


def test_make_run_agent_handler_accepts_parent_identity_kwargs() -> None:
  runner = _StubRunner()
  parent_session = GatewaySession(
    session_id="sess_parent",
    api_key_hash="hash",
    created_at=1,
    expires_at=2,
    user_id="operator",
    user_email="operator@example.com",
    auth_config={"provider": "anthropic", "billing_mode": "byok", "api_key": "key"},
  )
  handler = make_run_agent_handler(
    [runner],
    parent_session=parent_session,
    skill_loader=None,
    mcp_client=_StubMcpClient(),
    local_tool_handlers={},
    user_id="operator",
    user_email="operator@example.com",
    parent_user_id="alice",
    parent_user_email="alice@example.com",
  )

  result, error = _run(handler({"task": "Collect"}))

  assert error is None
  assert result == {"response": "ok"}
  assert runner.calls[0]["dispatcher"]._user_id == "alice"
  sub_session = runner.calls[0]["sub_session"]
  assert sub_session.user_id == "alice"
  assert sub_session.user_email == "alice@example.com"


def test_make_run_agent_handler_forwards_parent_turn_id_from_tool_context() -> None:
  runner = _StubRunner()
  handler = make_run_agent_handler(
    [runner],
    parent_session=GatewaySession(
      session_id="sess_parent",
      api_key_hash="hash",
      created_at=1,
      expires_at=2,
      user_id="alice",
      auth_config={"provider": "anthropic", "billing_mode": "byok", "api_key": "key"},
    ),
    skill_loader=None,
    mcp_client=_StubMcpClient(),
    local_tool_handlers={},
  )

  result, error = _run(
    handler(
      {"task": "Collect"},
      tool_ctx=ToolExecutionContext(
        tool_call_id="tool_run_agent_1",
        tool_name="run_agent",
        event_log=EventLog(),
      ),
    )
  )

  assert error is None
  assert result == {"response": "ok"}
  assert runner.calls[0]["parent_turn_id"] == "tool_run_agent_1"


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
  _write_skill(skills_dir, "research", _callable_skill("model: custom-model", body="Research."))
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


def test_make_run_agent_tool_def_includes_skill_descriptions_in_agent_param_only(tmp_path: Path) -> None:
  skills_dir = tmp_path / "skills"
  _write_skill(
    skills_dir,
    "alpha",
    _callable_skill("agent_description: Alpha reviews earnings.", body="Alpha."),
  )
  _write_skill(
    skills_dir,
    "beta",
    _callable_skill("agent_description: Beta reviews risk.", body="Beta."),
  )
  _write_skill(
    skills_dir,
    "hidden",
    "---\nagent_description: Hidden description.\n---\nHidden.",
  )

  tool_def = make_run_agent_tool_def(SkillLoader(skills_dir))
  agent_description = tool_def["input_schema"]["properties"]["agent"]["description"]

  assert "Available agents:" not in tool_def["description"]
  assert "Alpha reviews earnings." not in tool_def["description"]
  assert "Available agents:" in agent_description
  assert "- alpha: Alpha reviews earnings." in agent_description
  assert "- beta: Beta reviews risk." in agent_description
  assert "hidden" not in agent_description


def test_make_run_agent_tool_def_has_no_skill_suffix_without_skills() -> None:
  tool_def = make_run_agent_tool_def()

  assert "Available agents:" not in tool_def["description"]
  assert "One of:" not in tool_def["input_schema"]["properties"]["agent"]["description"]


def test_make_run_agent_handler_rejects_non_callable_named_skill(tmp_path: Path) -> None:
  skills_dir = tmp_path / "skills"
  _write_skill(skills_dir, "research", "---\nagent_description: Not callable.\n---\nResearch.")
  handler = make_run_agent_handler(
    [_StubRunner()],
    skill_loader=SkillLoader(skills_dir),
    mcp_client=_StubMcpClient(),
    local_tool_handlers={},
  )

  result, error = _run(handler({"agent": "research", "task": "Collect"}))

  assert result is None
  assert error == {
    "code": "invalid_skill",
    "message": "Agent 'research' is not callable. Choose a callable named agent or omit agent.",
  }


def test_make_run_agent_handler_background_registers_task(tmp_path: Path) -> None:
  runner = _StubRunner()
  handler = make_run_agent_handler(
    [runner],
    skill_loader=SkillLoader(tmp_path / "skills"),
    mcp_client=_StubMcpClient(),
    local_tool_handlers={"keep_tool": _dummy_tool},
  )

  result, error = _run(handler({"task": "Collect", "background": True}))

  assert error is None
  assert result == {"task_id": "bg_0", "status": "running"}
  assert runner.calls == []
  assert len(runner.background_calls) == 1
  assert runner.background_calls[0]["agent_name"] is None


def test_make_run_agent_handler_background_propagates_resumable_skill_flag(tmp_path: Path) -> None:
  skills_dir = tmp_path / "skills"
  _write_skill(skills_dir, "research", _callable_skill("resumable: true", body="Research."))
  runner = _StubRunner()
  handler = make_run_agent_handler(
    [runner],
    skill_loader=SkillLoader(skills_dir),
    mcp_client=_StubMcpClient(),
    local_tool_handlers={},
  )

  result, error = _run(handler({"agent": "research", "task": "Collect", "background": True}))

  assert error is None
  assert result == {"task_id": "bg_0", "status": "running"}
  assert runner.background_calls[0]["tool_input"]["resumable"] is True


def test_make_run_agent_handler_background_propagates_non_resumable_skill_flag(tmp_path: Path) -> None:
  skills_dir = tmp_path / "skills"
  _write_skill(skills_dir, "research", _callable_skill("resumable: false", body="Research."))
  runner = _StubRunner()
  handler = make_run_agent_handler(
    [runner],
    skill_loader=SkillLoader(skills_dir),
    mcp_client=_StubMcpClient(),
    local_tool_handlers={},
  )

  result, error = _run(handler({"agent": "research", "task": "Collect", "background": True}))

  assert error is None
  assert result == {"task_id": "bg_0", "status": "running"}
  assert runner.background_calls[0]["tool_input"]["resumable"] is False


def test_make_run_agent_handler_background_resumable_caller_override_wins(tmp_path: Path) -> None:
  skills_dir = tmp_path / "skills"
  _write_skill(skills_dir, "research", _callable_skill("resumable: true", body="Research."))
  runner = _StubRunner()
  handler = make_run_agent_handler(
    [runner],
    skill_loader=SkillLoader(skills_dir),
    mcp_client=_StubMcpClient(),
    local_tool_handlers={},
  )

  result, error = _run(
    handler({"agent": "research", "task": "Collect", "background": True, "resumable": False})
  )

  assert error is None
  assert result == {"task_id": "bg_0", "status": "running"}
  assert runner.background_calls[0]["tool_input"]["resumable"] is False


@pytest.mark.parametrize("raw_resumable", ["false", 0, None])
def test_make_run_agent_handler_background_rejects_non_bool_resumable_override(
  tmp_path: Path,
  raw_resumable: Any,
) -> None:
  skills_dir = tmp_path / "skills"
  _write_skill(skills_dir, "research", _callable_skill("resumable: true", body="Research."))
  runner = _StubRunner()
  handler = make_run_agent_handler(
    [runner],
    skill_loader=SkillLoader(skills_dir),
    mcp_client=_StubMcpClient(),
    local_tool_handlers={},
  )

  result, error = _run(
    handler({"agent": "research", "task": "Collect", "background": True, "resumable": raw_resumable})
  )

  assert result is None
  assert error == {
    "code": "invalid_input",
    "message": f"resumable must be a bool, got {type(raw_resumable).__name__}: {raw_resumable!r}",
  }
  assert runner.background_calls == []


def test_make_run_agent_handler_background_without_agent_does_not_enrich_resumable(tmp_path: Path) -> None:
  runner = _StubRunner()
  handler = make_run_agent_handler(
    [runner],
    skill_loader=SkillLoader(tmp_path / "skills"),
    mcp_client=_StubMcpClient(),
    local_tool_handlers={},
  )

  result, error = _run(handler({"task": "Collect", "background": True}))

  assert error is None
  assert result == {"task_id": "bg_0", "status": "running"}
  assert "resumable" not in runner.background_calls[0]["tool_input"]


def test_make_run_agent_handler_background_handler_forwards_task_entry(tmp_path: Path) -> None:
  runner = _StubRunner()
  handler = make_run_agent_handler(
    [runner],
    skill_loader=SkillLoader(tmp_path / "skills"),
    mcp_client=_StubMcpClient(),
    local_tool_handlers={"keep_tool": _dummy_tool},
  )

  result, error = _run(handler({"task": "Collect", "background": True}))

  assert error is None
  assert result == {"task_id": "bg_0", "status": "running"}
  background_handler = runner.background_calls[0]["handler"]
  task_entry = object()

  _run(background_handler({}, task_entry=task_entry, call_index=7))

  assert len(runner.calls) == 1
  assert runner.calls[0]["task_entry"] is task_entry
  assert runner.calls[0]["call_index"] == 7


def test_background_cleans_stale_output_file(tmp_path: Path) -> None:
  skills_dir = tmp_path / "skills"
  outputs_dir = tmp_path / "outputs"
  agent_name = "deep-research"
  _write_skill(
    skills_dir,
    agent_name,
    _callable_skill("persist_state: true"),
  )
  stale_path = outputs_dir / agent_name / f"{datetime.date.today().isoformat()}.md"
  stale_path.parent.mkdir(parents=True, exist_ok=True)
  stale_path.write_text("stale", encoding="utf-8")

  runner = _StubRunner()
  handler = make_run_agent_handler(
    [runner],
    skill_loader=SkillLoader(skills_dir),
    mcp_client=_StubMcpClient(),
    local_tool_handlers={},
    outputs_dir=outputs_dir,
  )

  result, error = _run(handler({"agent": agent_name, "task": "Collect", "background": True}))

  assert error is None
  assert result == {"task_id": "bg_0", "status": "running"}
  assert not stale_path.exists()
  assert len(runner.background_calls) == 1


def test_background_skips_cleanup_without_outputs_dir(tmp_path: Path) -> None:
  skills_dir = tmp_path / "skills"
  external_outputs_dir = tmp_path / "outputs"
  agent_name = "deep-research"
  _write_skill(
    skills_dir,
    agent_name,
    _callable_skill("persist_state: true"),
  )
  stale_path = external_outputs_dir / agent_name / f"{datetime.date.today().isoformat()}.md"
  stale_path.parent.mkdir(parents=True, exist_ok=True)
  stale_path.write_text("stale", encoding="utf-8")

  runner = _StubRunner()
  handler = make_run_agent_handler(
    [runner],
    skill_loader=SkillLoader(skills_dir),
    mcp_client=_StubMcpClient(),
    local_tool_handlers={},
  )

  result, error = _run(handler({"agent": agent_name, "task": "Collect", "background": True}))

  assert error is None
  assert result == {"task_id": "bg_0", "status": "running"}
  assert stale_path.exists()
  assert len(runner.background_calls) == 1


def test_background_cleans_for_non_persistent_skill_too(tmp_path: Path) -> None:
  skills_dir = tmp_path / "skills"
  outputs_dir = tmp_path / "outputs"
  agent_name = "deep-research"
  _write_skill(skills_dir, agent_name, _callable_skill())
  stale_path = outputs_dir / agent_name / f"{datetime.date.today().isoformat()}.md"
  stale_path.parent.mkdir(parents=True, exist_ok=True)
  stale_path.write_text("stale", encoding="utf-8")

  runner = _StubRunner()
  handler = make_run_agent_handler(
    [runner],
    skill_loader=SkillLoader(skills_dir),
    mcp_client=_StubMcpClient(),
    local_tool_handlers={},
    outputs_dir=outputs_dir,
  )

  result, error = _run(handler({"agent": agent_name, "task": "Collect", "background": True}))

  assert error is None
  assert result == {"task_id": "bg_0", "status": "running"}
  assert not stale_path.exists()
  assert len(runner.background_calls) == 1


def test_background_cleanup_returns_error_on_oserror(
  tmp_path: Path,
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  skills_dir = tmp_path / "skills"
  outputs_dir = tmp_path / "outputs"
  agent_name = "deep-research"
  _write_skill(
    skills_dir,
    agent_name,
    _callable_skill("persist_state: true"),
  )
  stale_path = outputs_dir / agent_name / f"{datetime.date.today().isoformat()}.md"
  stale_path.parent.mkdir(parents=True, exist_ok=True)
  stale_path.write_text("stale", encoding="utf-8")

  original_unlink = Path.unlink

  def _broken_unlink(self: Path, *, missing_ok: bool = False) -> None:
    if self == stale_path:
      raise OSError("boom")
    original_unlink(self, missing_ok=missing_ok)

  monkeypatch.setattr(Path, "unlink", _broken_unlink)

  runner = _StubRunner()
  handler = make_run_agent_handler(
    [runner],
    skill_loader=SkillLoader(skills_dir),
    mcp_client=_StubMcpClient(),
    local_tool_handlers={},
    outputs_dir=outputs_dir,
  )

  result, error = _run(handler({"agent": agent_name, "task": "Collect", "background": True}))

  assert result is None
  assert error == {
    "code": "file_cleanup_failed",
    "message": f"Failed to clean stale output {stale_path}: boom",
  }
  assert len(runner.background_calls) == 0


def test_make_get_background_result_handler_proxies_to_runner() -> None:
  runner = _StubRunner()
  handler = make_get_background_result_handler([runner])

  result, error = _run(handler({"task_id": "bg_7"}))

  assert error is None
  assert result == {"task_id": "bg_7", "status": "completed"}


def test_make_get_background_result_tool_def_has_expected_schema() -> None:
  tool_def = make_get_background_result_tool_def()

  assert tool_def["name"] == "get_background_result"
  assert set(tool_def["input_schema"]["properties"]) == {"task_id", "wait", "timeout"}
  assert tool_def["input_schema"]["required"] == ["task_id"]
