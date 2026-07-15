# ruff: noqa: E402

import asyncio
import datetime
import re
import sys
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[3]
PKG_DIR = ROOT / "packages" / "agent-gateway"
if str(PKG_DIR) not in sys.path:
  sys.path.insert(0, str(PKG_DIR))

from agent_gateway.skills import SkillLoader, SkillProfile, SkillStateStore
from agent_gateway.commercial_work_start import CommercialWorkStartContext
from agent_gateway import EventLog
from agent_gateway.html_artifact_store import read_html_artifact_content, read_html_artifact_sidecar
from agent_gateway.session import GatewaySession
import agent_gateway.sub_agent as sub_agent_module
import agent_gateway.sub_agent_background_result as sub_agent_background_result
import agent_gateway.sub_agent_helpers as sub_agent_helpers
import agent_gateway.sub_agent_skill_events as sub_agent_skill_events
import agent_gateway.sub_agent_tool_definitions as sub_agent_tool_definitions
from agent_gateway.sub_agent import (
  _DEFAULT_EXCLUDED_TOOLS,
  DEFAULT_SUB_AGENT_TIMEOUT_SECONDS,
  _DEFAULT_SYSTEM_PROMPT_TEMPLATE,
  _SKILL_SYSTEM_PROMPT_TEMPLATE,
  make_get_background_result_handler,
  make_get_background_result_tool_def,
  make_run_agent_handler,
  make_run_agent_tool_def,
)
from agent_gateway.tool_dispatcher import ToolExecutionContext

_UNRESOLVED_BLOCK_RE = re.compile(r"\{\{[A-Z][A-Z0-9_]*\}\}")


def test_sub_agent_helper_exports_are_parent_aliases() -> None:
  helper_names = (
    "_DEFAULT_EXCLUDED_TOOLS",
    "DEFAULT_SUB_AGENT_TIMEOUT_SECONDS",
    "_ARTIFACT_EMIT_TOOLS",
    "_RESEARCH_FILE_ID_RE",
    "ExcludedToolsResolver",
    "NeedsApprovalResolver",
    "MutationModeExclusionsApplier",
    "_SKILL_SYSTEM_PROMPT_TEMPLATE",
    "_DEFAULT_SYSTEM_PROMPT_TEMPLATE",
    "_RUN_AGENT_DESCRIPTION",
    "_RESUME_AGENT_DESCRIPTION",
    "_entry_child_budget_usd",
    "_CONTEXT_TICKER_RE",
    "_TICKER_STOPWORDS",
    "_artifact_storage_user_id",
    "_render_agent_param_description",
    "_extract_ticker_from_task",
    "_extract_research_file_id_from_task",
    "_optional_research_file_id",
    "_message_content_text",
    "_extract_ticker_from_resume_messages",
    "_extract_research_file_id_from_resume_messages",
    "_html_artifact_ticker",
    "_html_artifact_scope",
    "_dashboard_artifact_ticker",
    "_dashboard_artifact_scope",
    "_skill_html_excluded_tools",
    "_install_emit_html_artifact_handler",
    "_install_emit_dashboard_artifact_handler",
    "_skill_extra_excluded_tool_names",
    "make_run_agent_tool_def",
    "make_get_background_result_tool_def",
    "make_resume_tool_def",
    "make_send_message_tool_def",
  )

  for name in helper_names:
    assert getattr(sub_agent_module, name) is getattr(sub_agent_helpers, name)

  assert (
    sub_agent_module.make_get_background_result_handler
    is sub_agent_background_result.make_get_background_result_handler
  )


@pytest.mark.parametrize("ticker", ["AXIA7", "JBSS32", "AURA33", "TAEE11", "AAPL"])
def test_lh16_context_ticker_discovery_positive_vectors(ticker: str) -> None:
  assert sub_agent_helpers._extract_ticker_from_task(f"Analyze {ticker} now") == ticker


@pytest.mark.parametrize(
  "token",
  ["AXIA1", "AXIA2", "DGS10", "E2E", "F128", "FY2026", "FY26E", "2026E", "2026Q1", "10KB"],
)
def test_lh16_context_ticker_discovery_exclusion_vectors(token: str) -> None:
  assert sub_agent_helpers._extract_ticker_from_task(token) == ""


def test_sub_agent_tool_definition_wrappers_preserve_parent_behavior(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  class _RunnerWithCatalog:
    def _get_tool_definitions(self) -> list[dict[str, Any]]:
      return [
        {"name": "keep_tool", "input_schema": {"properties": {"value": {"type": "string"}}}},
        {"name": "blocked_tool"},
      ]

  extra_definitions = [
    {"name": "extra_tool", "input_schema": {"properties": {"items": []}}},
    {"name": "keep_tool", "description": "duplicate should not be appended"},
  ]
  getter = sub_agent_module._child_tool_definitions_getter(
    runner=_RunnerWithCatalog(),
    mcp_client=object(),
    excluded_tools={"blocked_tool"},
    extra_tool_definitions=extra_definitions,
  )

  assert getter is not None
  definitions = getter()
  assert [definition["name"] for definition in definitions] == ["keep_tool", "extra_tool"]
  definitions[1]["input_schema"]["properties"]["items"].append("mutated")
  assert getter()[1]["input_schema"]["properties"]["items"] == []

  monkeypatch.setattr(sub_agent_module, "_ARTIFACT_EMIT_TOOLS", frozenset({"emit_html_artifact"}))
  wrapper_definitions = sub_agent_module._artifact_emit_tool_definitions(
    {"emit_dashboard_artifact", "emit_html_artifact"},
  )
  direct_definitions = sub_agent_tool_definitions.artifact_emit_tool_definitions(
    {"emit_html_artifact"},
    artifact_emit_tools={"emit_html_artifact"},
  )
  assert [definition["name"] for definition in wrapper_definitions] == ["emit_html_artifact"]
  assert wrapper_definitions == direct_definitions
  wrapper_definitions[0]["name"] = "mutated"
  assert sub_agent_module._artifact_emit_tool_definitions({"emit_html_artifact"})[0]["name"] == (
    "emit_html_artifact"
  )


def test_skill_run_event_emitter_emits_started_once_and_captures_result() -> None:
  sub_log = EventLog()
  parent_events: list[dict[str, Any]] = []
  profile = type("Profile", (), {"name": "html-research"})()
  emitter = sub_agent_skill_events.SkillRunEventEmitter(
    skill_run_id="skill-1",
    profile=profile,
    context_ticker="pcty",
    event_log_getter=lambda: sub_log,
    tool_ctx=type("Ctx", (), {"emit": parent_events.append})(),
    ticker_fn=lambda _profile, ticker: ticker.upper(),
    scope_fn=lambda _profile, _ticker: "ticker",
    time_fn=lambda: 123.0,
  )

  emitter.emit_started()
  emitter.emit_started()
  emitter.emit_result_captured({"response": "done"}, None)

  events = [entry.event for entry in sub_log.entries]
  assert [event["type"] for event in events] == ["skill_run_started", "skill_result_captured"]
  assert events == parent_events
  assert events[0]["skill_run_id"] == "skill-1"
  assert events[0]["skill"] == "html-research"
  assert events[0]["ticker"] == "PCTY"
  assert events[0]["ts"] == 123.0
  assert events[1]["status"] == "success"


def test_skill_run_event_emitter_captures_existing_artifact_events() -> None:
  sub_log = EventLog()
  profile = type("Profile", (), {"name": "html-research"})()
  emitter = sub_agent_skill_events.SkillRunEventEmitter(
    skill_run_id="skill-artifact",
    profile=profile,
    context_ticker="",
    event_log_getter=lambda: sub_log,
    tool_ctx=object(),
    ticker_fn=lambda _profile, _ticker: None,
    scope_fn=lambda _profile, _ticker: "portfolio",
    time_fn=lambda: 123.0,
  )
  artifact_event = {
    "type": "artifact_ready",
    "skill_run_id": "skill-artifact",
    "ticker": "PCTY",
    "skill": "_html",
    "artifact_id": "art-1",
    "artifact_path": "artifacts/_html/art-1.json",
    "contract_name": "HtmlArtifact",
    "data_source": "live",
    "ts": 124.0,
  }

  emitter.emit_started()
  emitter.emit_parent_event(artifact_event)
  emitter.emit_result_captured({"response": "done"}, None)

  captured = sub_log.entries[-1].event
  assert captured["type"] == "skill_result_captured"
  assert captured["ticker"] == "PCTY"
  assert captured["artifact_events"] == [artifact_event]
  assert captured["artifact_refs"] == ["artifacts/_html/art-1.json"]


def test_skill_run_event_emitter_preserves_parent_emit_before_log_exists() -> None:
  parent_events: list[dict[str, Any]] = []
  profile = type("Profile", (), {"name": "resume-skill"})()

  def missing_log() -> Any:
    raise NameError("sub_log")

  emitter = sub_agent_skill_events.SkillRunEventEmitter(
    skill_run_id="skill-2",
    profile=profile,
    context_ticker="",
    event_log_getter=missing_log,
    tool_ctx=type("Ctx", (), {"emit": parent_events.append})(),
    ticker_fn=lambda _profile, _ticker: None,
    scope_fn=lambda _profile, _ticker: "portfolio",
    time_fn=lambda: 456.0,
  )

  emitter.emit_started()

  assert [event["type"] for event in parent_events] == ["skill_run_started"]
  assert parent_events[0]["scope"] == "portfolio"


def test_skill_html_excluded_tools_rejects_malformed_extra_exclusions() -> None:
  profile = SkillProfile(
    name="bad-extra",
    system_prompt="",
    agent_callable=True,
    agent_description="Bad extra exclusions.",
  )
  profile.extra_excluded_tools = "emit_html_artifact"  # type: ignore[assignment]

  with pytest.raises(ValueError, match="extra_excluded_tools must be a list of tool names"):
    sub_agent_module._skill_html_excluded_tools(set(), skill_profile=profile)


def test_skill_html_excluded_tools_rejects_null_extra_exclusions() -> None:
  profile = SkillProfile(
    name="bad-extra",
    system_prompt="",
    agent_callable=True,
    agent_description="Bad extra exclusions.",
  )
  profile.extra_excluded_tools = ["emit_html_artifact", None]  # type: ignore[list-item]

  with pytest.raises(ValueError, match="extra_excluded_tools entries must be strings"):
    sub_agent_module._skill_html_excluded_tools(set(), skill_profile=profile)


def _run(coro):
  return asyncio.run(coro)


async def _dummy_tool(_tool_input, **_kwargs):
  return {"ok": True}, None


class _StubMcpClient:
  def is_mcp_tool(self, _name: str) -> bool:
    return False

  async def call_tool(self, _name: str, _tool_input: dict):
    raise AssertionError("unexpected MCP tool dispatch")


class _CatalogMcpClient(_StubMcpClient):
  def get_tool_definitions(self) -> list[dict[str, Any]]:
    return []


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


class _StaticSkillLoader:
  def __init__(self, skills_dir: Path, profile: SkillProfile) -> None:
    self.skills_dir = skills_dir
    self._profile = profile

  def load(self, _name: str) -> SkillProfile:
    return self._profile


def _callable_skill(frontmatter: str = "", body: str = "Use multiple sources.") -> str:
  frontmatter = frontmatter.strip()
  lines = ["---", "agent_callable: true", "agent_description: Callable test skill."]
  if frontmatter:
    lines.extend(frontmatter.splitlines())
  lines.extend(["---", body])
  return "\n".join(lines)


def test_make_run_agent_handler_installs_emit_html_artifact_for_named_skill(
  tmp_path: Path,
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  import memory

  monkeypatch.setenv("USER_DATA_DIR", str(tmp_path / "data"))
  memory.set_memory_store_factory(None)
  try:
    skills_dir = tmp_path / "skills"
    _write_skill(skills_dir, "html-research", _callable_skill("scope: ticker", body="Research."))
    parent_session = GatewaySession(
      session_id="sess_parent",
      api_key_hash="hash",
      created_at=1,
      expires_at=2,
      user_id="alice",
      user_email="alice@example.com",
      auth_config={"provider": "anthropic", "billing_mode": "byok", "api_key": "key"},
    )
    runner = _StubRunner()
    parent_log = EventLog()
    handler = make_run_agent_handler(
      [runner],
      parent_session=parent_session,
      skill_loader=SkillLoader(skills_dir),
      mcp_client=_StubMcpClient(),
      local_tool_handlers={},
      excluded_tools={"emit_html_artifact"},
      default_model="claude-sonnet-4-6",
      allowed_models={"claude-sonnet-4-6"},
    )

    result, error = _run(
      handler(
        {"agent": "html-research", "task": "Analyze PCTY with rich visuals"},
        tool_ctx=ToolExecutionContext(
          tool_call_id="tool_run_agent_1",
          tool_name="run_agent",
          event_log=parent_log,
        ),
      )
    )
    assert error is None
    assert result == {"response": "ok"}
    dispatcher = runner.calls[0]["dispatcher"]
    assert "emit_html_artifact" not in runner.calls[0]["excluded_tools"]
    advertised_tools = {tool["name"] for tool in dispatcher._get_tool_definitions()}
    assert "emit_html_artifact" in advertised_tools
    assert "emit_dashboard_artifact" in advertised_tools

    emit_result, emit_error = _run(
      dispatcher.dispatch(
        "tool_html_1",
        "emit_html_artifact",
        {
          "title": "PCTY Historical Coincidences",
          "purpose": "exploration",
          "summary": "Timeline of PCTY coincidences.",
          "html": "<section><h1>PCTY</h1></section>",
          "copy_as_prompt": "Analyze PCTY",
          "copy_as_markdown": "## PCTY",
          "copy_as_json": {"ticker": "PCTY"},
          "sources": [],
        },
      )
    )

    assert emit_error is None
    assert emit_result is not None
    artifact_id = emit_result["artifact_id"]
    workspace_dir = memory.get_workspace_dir("alice")
    sidecar = read_html_artifact_sidecar(workspace_dir, artifact_id)
    assert sidecar is not None
    assert sidecar.title == "PCTY Historical Coincidences"
    assert sidecar.ticker == "PCTY"
    assert sidecar.source_skill == "html-research"
    assert sidecar.exports.copy_as_json == {"ticker": "PCTY"}
    assert read_html_artifact_content(workspace_dir, artifact_id) == "<section><h1>PCTY</h1></section>"

    events = [entry.event for entry in parent_log.entries]
    assert [event["type"] for event in events] == [
      "skill_run_started",
      "skill_result_captured",
      "artifact_ready",
    ]
    assert events[1]["skill"] == "html-research"
    assert events[1]["artifact_refs"] == []
    ready = events[-1]
    assert ready["artifact_id"] == artifact_id
    assert ready["ticker"] == "PCTY"
    assert ready["skill"] == "_html"
    assert ready["artifact_path"] == f"artifacts/_html/{artifact_id}.json"
    assert ready["binary_artifact_path"] == f"artifacts/_html/{artifact_id}.html"
    assert ready["contract_name"] == "HtmlArtifact"
  finally:
    memory.set_memory_store_factory(None)


def test_emit_html_artifact_failure_emits_tool_write_failed(
  tmp_path: Path,
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  import memory

  monkeypatch.setenv("USER_DATA_DIR", str(tmp_path / "data"))
  memory.set_memory_store_factory(None)
  try:
    skills_dir = tmp_path / "skills"
    _write_skill(skills_dir, "portfolio-report", _callable_skill("scope: portfolio", body="Report."))
    parent_session = GatewaySession(
      session_id="sess_parent",
      api_key_hash="hash",
      created_at=1,
      expires_at=2,
      user_id="alice",
      user_email="alice@example.com",
      auth_config={"provider": "anthropic", "billing_mode": "byok", "api_key": "key"},
    )
    runner = _StubRunner()
    parent_log = EventLog()
    handler = make_run_agent_handler(
      [runner],
      parent_session=parent_session,
      skill_loader=SkillLoader(skills_dir),
      mcp_client=_StubMcpClient(),
      local_tool_handlers={},
      default_model="claude-sonnet-4-6",
      allowed_models={"claude-sonnet-4-6"},
    )

    _run(
      handler(
        {"agent": "portfolio-report", "task": "Analyze portfolio"},
        tool_ctx=ToolExecutionContext(
          tool_call_id="tool_run_agent_1",
          tool_name="run_agent",
          event_log=parent_log,
        ),
      )
    )
    dispatcher = runner.calls[0]["dispatcher"]

    emit_result, emit_error = _run(
      dispatcher.dispatch(
        "tool_html_bad",
        "emit_html_artifact",
        {
          "title": "Broken report",
          "purpose": "report",
          "summary": "Empty html field.",
          "html": "",
          "sources": [],
        },
      )
    )

    assert emit_result is None
    assert emit_error is not None
    assert emit_error["code"] == "internal_error"
    assert "non-empty string" in emit_error["message"]
    events = [entry.event for entry in parent_log.entries]
    assert [event["type"] for event in events] == [
      "skill_run_started",
      "skill_result_captured",
      "artifact_failed",
    ]
    assert events[0]["ticker"] is None
    assert events[0]["scope"] == "portfolio"
    failed = events[-1]
    assert failed["ticker"] is None
    assert failed["skill"] == "_html"
    assert failed["error_code"] == "tool_write_failed"
    assert failed["source_path"] is None
    assert failed["tool_call_id"] == "tool_html_bad"
    assert not (memory.get_workspace_dir("alice") / "artifacts" / "_html").exists()
  finally:
    memory.set_memory_store_factory(None)


def test_emit_html_artifact_uses_risk_user_storage_workspace(
  tmp_path: Path,
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  import memory

  monkeypatch.setenv("USER_DATA_DIR", str(tmp_path / "data"))
  memory.set_memory_store_factory(None)
  try:
    skills_dir = tmp_path / "skills"
    _write_skill(skills_dir, "html-research", _callable_skill("scope: ticker", body="Research."))
    parent_session = GatewaySession(
      session_id="sess_parent",
      api_key_hash="hash",
      created_at=1,
      expires_at=2,
      user_id="alice-slug",
      user_email="alice@example.com",
      risk_user_id=42,
      auth_config={"provider": "anthropic", "billing_mode": "byok", "api_key": "key"},
    )
    runner = _StubRunner()
    handler = make_run_agent_handler(
      [runner],
      parent_session=parent_session,
      skill_loader=SkillLoader(skills_dir),
      mcp_client=_StubMcpClient(),
      local_tool_handlers={},
      default_model="claude-sonnet-4-6",
      allowed_models={"claude-sonnet-4-6"},
    )

    _run(handler({"agent": "html-research", "task": "Analyze PCTY"}))
    dispatcher = runner.calls[0]["dispatcher"]
    emit_result, emit_error = _run(
      dispatcher.dispatch(
        "tool_html_1",
        "emit_html_artifact",
        {
          "title": "PCTY View",
          "purpose": "exploration",
          "summary": "PCTY visual analysis.",
          "html": "<section><h1>PCTY</h1></section>",
          "sources": [],
        },
      )
    )

    assert emit_error is None
    artifact_id = emit_result["artifact_id"]
    assert read_html_artifact_sidecar(memory.get_workspace_dir("42"), artifact_id) is not None
    assert read_html_artifact_sidecar(memory.get_workspace_dir("alice-slug"), artifact_id) is None
  finally:
    memory.set_memory_store_factory(None)


def test_make_run_agent_handler_loads_skill_profile_and_filters_tools(tmp_path: Path) -> None:
  skills_dir = tmp_path / "skills"
  _write_skill(
    skills_dir,
    "deep-research",
    _callable_skill("model: claude-opus-4-6\nmax_turns: 7\ntimeout: 12.5\nmax_tokens: 12000"),
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
  assert call["max_tokens"] == 12000
  assert call["call_index"] == 3
  assert dispatcher._local["keep_tool"] is keep_tool
  assert "drop_tool" not in dispatcher._local
  assert "run_agent" not in dispatcher._local
  assert "emit_html_artifact" in dispatcher._local
  assert dispatcher._needs_approval("keep_tool", {}, "") is False
  assert dispatcher._session_id == runner._full_session_id


def test_run_agent_rejects_commercial_provider_mismatch_before_spawn(tmp_path: Path) -> None:
  runner = _StubRunner()
  resolver_calls: list[str] = []
  work_start = CommercialWorkStartContext(
    claim=object(),
    authorization=type("Authorization", (), {"provider": "anthropic"})(),
    consumption=object(),
  )
  handler = make_run_agent_handler(
    [runner],
    skill_loader=SkillLoader(tmp_path / "skills"),
    mcp_client=_StubMcpClient(),
    local_tool_handlers={},
    default_model="gpt-test",
    provider_resolver=lambda provider: resolver_calls.append(provider) or type(
      "ResolvedProvider",
      (),
      {"allowed_models": {"gpt-test"}, "default_model": "gpt-test", "provider": object()},
    )(),
    commercial_work_start=work_start,
  )

  result, error = _run(
    handler({"task": "Analyze filings", "provider": "openai"})
  )

  assert result is None
  assert error["code"] == "commercial_child_provider_mismatch"
  assert resolver_calls == []
  assert runner.calls == []


def test_run_agent_threads_commercial_controls_to_child_dispatcher(tmp_path: Path) -> None:
  runner = _StubRunner()
  work_start = CommercialWorkStartContext(
    claim=object(),
    authorization=type("Authorization", (), {"provider": "anthropic"})(),
    consumption=object(),
  )
  irreversible_recheck = object()
  mcp_servers = frozenset({"portfolio-trades-mcp"})
  handler = make_run_agent_handler(
    [runner],
    skill_loader=SkillLoader(tmp_path / "skills"),
    mcp_client=_StubMcpClient(),
    local_tool_handlers={},
    default_model="claude-sonnet-4-6",
    allowed_models={"claude-sonnet-4-6"},
    commercial_work_start=work_start,
    commercial_irreversible_recheck=irreversible_recheck,
    commercial_mcp_servers=mcp_servers,
  )

  result, error = _run(handler({"task": "Analyze filings"}))

  assert error is None
  assert result == {"response": "ok"}
  dispatcher = runner.calls[0]["dispatcher"]
  assert dispatcher._commercial_work_start is work_start
  assert dispatcher._commercial_irreversible_recheck is irreversible_recheck
  assert dispatcher._commercial_mcp_servers is mcp_servers


def test_make_run_agent_handler_child_dispatcher_validates_local_tool_schema(tmp_path: Path) -> None:
  skills_dir = tmp_path / "skills"
  _write_skill(skills_dir, "deep-research", _callable_skill())
  calls: list[dict[str, Any]] = []

  async def _structured_write(tool_input: dict[str, Any], **_kwargs):
    calls.append(dict(tool_input))
    return {"ok": True}, None

  tool_def = {
    "name": "structured_write",
    "description": "test",
    "input_schema": {
      "type": "object",
      "properties": {"judgment": {"type": "object"}},
      "required": ["judgment"],
      "additionalProperties": False,
    },
  }

  class _SchemaGuardRunner(_StubRunner):
    def _get_tool_definitions(self) -> list[dict[str, Any]]:
      return [tool_def]

    async def spawn_sub_agent(self, task: str, **kwargs):
      _ = task
      result, error = await kwargs["dispatcher"].dispatch(
        "child-call-1",
        "structured_write",
        {},
      )
      return {"child_result": result, "child_error": error}, None

  runner = _SchemaGuardRunner()
  handler = make_run_agent_handler(
    [runner],
    skill_loader=SkillLoader(skills_dir),
    mcp_client=_StubMcpClient(),
    local_tool_handlers={"structured_write": _structured_write},
    default_model="claude-sonnet-4-6",
    allowed_models={"claude-sonnet-4-6"},
  )

  result, error = _run(handler({"agent": "deep-research", "task": "Analyze filings"}))

  assert error is None
  assert result["child_result"] is None
  assert result["child_error"]["code"] == "invalid_tool_input_schema"
  assert result["child_error"]["details"]["missing"] == ["judgment"]
  assert calls == []


def test_make_run_agent_handler_rejects_malformed_profile_extra_exclusions(tmp_path: Path) -> None:
  skills_dir = tmp_path / "skills"
  skills_dir.mkdir()
  profile = SkillProfile(
    name="bad-extra",
    system_prompt="Use multiple sources.",
    agent_callable=True,
    agent_description="Bad extra exclusions.",
  )
  profile.extra_excluded_tools = "drop_tool"  # type: ignore[assignment]
  runner = _StubRunner()
  handler = make_run_agent_handler(
    [runner],
    skill_loader=_StaticSkillLoader(skills_dir, profile),  # type: ignore[arg-type]
    mcp_client=_StubMcpClient(),
    local_tool_handlers={"drop_tool": _dummy_tool},
    default_model="claude-sonnet-4-6",
    allowed_models={"claude-sonnet-4-6"},
  )

  result, error = _run(handler({"agent": "bad-extra", "task": "Analyze filings"}))

  assert result is None
  assert error == {
    "code": "invalid_skill_config",
    "message": "Skill 'bad-extra' extra_excluded_tools must be a list of tool names",
  }
  assert runner.calls == []


def test_make_run_agent_handler_resolves_skill_blocks_before_spawn(tmp_path: Path) -> None:
  skills_dir = tmp_path / "skills"
  blocks_dir = skills_dir / "_blocks"
  blocks_dir.mkdir(parents=True)
  (blocks_dir / "citation-contract.md").write_text(
    "Resolved citation contract.\nSecond line stays verbatim.\n",
    encoding="utf-8",
  )
  _write_skill(
    skills_dir,
    "blocked-research",
    _callable_skill(body="Use sources.\n{{CITATION_CONTRACT}}\nReport findings."),
  )
  assert "{{CITATION_CONTRACT}}" in SkillLoader(skills_dir).load("blocked-research").system_prompt
  runner = _StubRunner()
  handler = make_run_agent_handler(
    [runner],
    skill_loader=SkillLoader(skills_dir),
    mcp_client=_StubMcpClient(),
    local_tool_handlers={},
    default_model="claude-sonnet-4-6",
    allowed_models={"claude-sonnet-4-6"},
  )

  result, error = _run(handler({"agent": "blocked-research", "task": "Analyze filings"}))

  assert error is None
  assert result == {"response": "ok"}
  prompt = runner.calls[0]["system_prompt"]
  assert "Resolved citation contract.\nSecond line stays verbatim.\n" in prompt
  assert not _UNRESOLVED_BLOCK_RE.search(prompt)


def test_make_run_agent_handler_uses_anonymous_defaults_for_blank_agent(
  tmp_path: Path,
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  monkeypatch.delenv("SUB_AGENT_DEFAULT_MODEL", raising=False)
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


def test_make_run_agent_handler_does_not_advertise_artifact_stub_for_generic_agent(
  tmp_path: Path,
) -> None:
  runner = _StubRunner()
  handler = make_run_agent_handler(
    [runner],
    skill_loader=SkillLoader(tmp_path / "skills"),
    mcp_client=_CatalogMcpClient(),
    local_tool_handlers={"emit_html_artifact": _dummy_tool},
    default_model="claude-opus-4-6",
  )

  result, error = _run(handler({"agent": "   ", "task": "Quick question"}))

  assert error is None
  assert result == {"response": "ok"}
  dispatcher = runner.calls[0]["dispatcher"]
  assert "emit_html_artifact" in dispatcher._local
  assert dispatcher._get_tool_definitions() == []
  emit_result, emit_error = _run(
    dispatcher.dispatch(
      "tool_html_generic",
      "emit_html_artifact",
      {
        "title": "Generic",
        "purpose": "exploration",
        "summary": "Should not be advertised.",
        "html": "<main>generic</main>",
      },
    )
  )
  assert emit_result is None
  assert emit_error is not None
  assert emit_error["code"] == "tool_not_advertised"


def test_make_run_agent_handler_uses_sub_agent_default_model_knob(
  tmp_path: Path,
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  runner = _StubRunner()
  monkeypatch.setenv("SUB_AGENT_DEFAULT_MODEL", "claude-opus-4-8")
  handler = make_run_agent_handler(
    [runner],
    skill_loader=SkillLoader(tmp_path / "skills"),
    mcp_client=_StubMcpClient(),
    local_tool_handlers={},
    default_model="claude-sonnet-4-6",
    allowed_models={"claude-sonnet-4-6", "claude-opus-4-8"},
  )

  result, error = _run(handler({"task": "Quick question"}))

  assert error is None
  assert result == {"response": "ok"}
  assert runner.calls[0]["model"] == "claude-opus-4-8"


def test_make_run_agent_handler_skill_pin_beats_sub_agent_default_model_knob(
  tmp_path: Path,
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  skills_dir = tmp_path / "skills"
  _write_skill(skills_dir, "deep-research", _callable_skill("model: claude-opus-4-6"))
  runner = _StubRunner()
  monkeypatch.setenv("SUB_AGENT_DEFAULT_MODEL", "claude-opus-4-8")
  handler = make_run_agent_handler(
    [runner],
    skill_loader=SkillLoader(skills_dir),
    mcp_client=_StubMcpClient(),
    local_tool_handlers={},
    default_model="claude-sonnet-4-6",
    allowed_models={"claude-sonnet-4-6", "claude-opus-4-6", "claude-opus-4-8"},
  )

  result, error = _run(handler({"agent": "deep-research", "task": "Analyze filings"}))

  assert error is None
  assert result == {"response": "ok"}
  assert runner.calls[0]["model"] == "claude-opus-4-6"


def test_make_run_agent_handler_default_timeout_is_finite() -> None:
  # ACUI-1: timeout=None let a wedged sub-agent hold the parent turn open
  # forever; the default must be a finite bound (profiles still override).
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
  assert runner.calls[0]["timeout"] == DEFAULT_SUB_AGENT_TIMEOUT_SECONDS


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


def test_make_run_agent_handler_provider_error_precedes_skill_block_resolution(tmp_path: Path) -> None:
  skills_dir = tmp_path / "skills"
  _write_skill(
    skills_dir,
    "research",
    _callable_skill("provider: openai", body="Research.\n{{MISSING_BLOCK}}\n"),
  )
  handler = make_run_agent_handler(
    [_StubRunner()],
    skill_loader=SkillLoader(skills_dir),
    mcp_client=_StubMcpClient(),
    local_tool_handlers={},
  )

  result, error = _run(handler({"agent": "research", "task": "Collect"}))

  assert result is None
  assert error == {
    "code": "provider_not_supported",
    "message": "Provider 'openai' requested but no provider_resolver configured",
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
  parent_log = EventLog()
  handler = make_run_agent_handler(
    [runner],
    skill_loader=SkillLoader(skills_dir),
    mcp_client=_StubMcpClient(),
    local_tool_handlers={},
  )

  result, error = _run(
    handler(
      {"agent": "research", "task": "Collect", "background": True},
      tool_ctx=ToolExecutionContext(
        tool_call_id="tool_run_agent_1",
        tool_name="run_agent",
        event_log=parent_log,
      ),
    )
  )

  assert error is None
  assert result == {"task_id": "bg_0", "status": "running"}
  assert runner.background_calls[0]["tool_input"]["resumable"] is True
  on_complete = runner.background_calls[0]["on_complete"]
  assert callable(on_complete)

  bg_task = type("BgTask", (), {"result": {"response": "done"}, "error": None})()
  _run(on_complete(bg_task))

  events = [entry.event for entry in parent_log.entries]
  assert [event["type"] for event in events] == ["skill_run_started", "skill_result_captured"]
  assert events[-1]["skill"] == "research"
  assert events[-1]["status"] == "success"


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


def test_persistent_named_skill_injects_and_updates_state(tmp_path: Path) -> None:
  skills_dir = tmp_path / "skills"
  agent_name = "deep-research"
  _write_skill(
    skills_dir,
    agent_name,
    _callable_skill(
      "persist_state: true\nversion: 1.2.3",
      body="Use prior state when relevant.",
    ),
  )
  store = SkillStateStore(tmp_path / "skill_state.json")
  store.set(agent_name, {"existing": "keep", "run_count": 2})

  class _StateRunner(_StubRunner):
    async def spawn_sub_agent(self, task: str, **kwargs):
      self.calls.append({"task": task, **kwargs})
      return {
        "response": (
          "Done.\n\n"
          "## STATE_UPDATE_JSON\n"
          "```json\n"
          "{\"alerts\":[\"Watch earnings\"]}\n"
          "```"
        )
      }, None

  runner = _StateRunner()
  handler = make_run_agent_handler(
    [runner],
    skill_loader=SkillLoader(skills_dir),
    mcp_client=_StubMcpClient(),
    local_tool_handlers={},
    default_model="claude-test",
    allowed_models={"claude-test"},
    skill_state_store=store,
  )

  result, error = _run(handler({"agent": agent_name, "task": "Collect"}))

  assert error is None
  assert result is not None
  prompt = runner.calls[0]["system_prompt"]
  assert "## Persisted Skill State" in prompt
  assert '"existing": "keep"' in prompt
  state = store.get(agent_name)
  assert state["existing"] == "keep"
  assert state["alerts"] == ["Watch earnings"]
  assert state["model"] == "claude-test"
  assert state["version"] == "1.2.3"
  assert state["run_count"] == 3
  assert "last_run" in state


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
