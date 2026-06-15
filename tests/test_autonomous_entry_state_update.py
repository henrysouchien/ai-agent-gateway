import asyncio
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
  sys.path.insert(0, str(ROOT))
API_DIR = ROOT / "api"
if str(API_DIR) not in sys.path:
  sys.path.insert(0, str(API_DIR))
PKG_DIR = ROOT / "packages" / "agent-gateway"
if str(PKG_DIR) not in sys.path:
  sys.path.insert(0, str(PKG_DIR))

from agent_gateway import AgentSessionLog, EventLog
# entry.py calls validate_product_id_or_raise() at import (app fail-fast).
# Provide a valid placeholder so the module imports in CI / fresh checkouts
# where PRODUCT_ID (a deployment env var) is unset. setdefault never overrides
# a real value; nothing in the suite asserts PRODUCT_ID-unset behavior.
os.environ.setdefault("PRODUCT_ID", "hank-test")
from api.agent.autonomous import entry as autonomous_entry


@pytest.fixture(autouse=True)
def _autonomous_identity(monkeypatch: pytest.MonkeyPatch) -> None:
  monkeypatch.setenv("AUTONOMOUS_USER_ID", "henry")
  monkeypatch.setenv("AUTONOMOUS_USER_EMAIL", "hc@henrychien.com")


def _run(coro):
  return asyncio.run(coro)


def _analyst_run_once_profile(**overrides: Any) -> SimpleNamespace:
  profile = SimpleNamespace(
    name="analyst",
    run_once_session_id_template="{profile}:{today}",
    briefing_file_template="analyst/{date}.md",
    run_once_excluded_tools=None,
    excluded_tools=set(),
    model="claude-sonnet-4-6",
    max_turns=5,
    timeout_seconds=60.0,
    per_turn_timeout=None,
    max_tokens=16000,
    client_timeout=30.0,
    max_budget_usd=2.0,
    compaction_instructions=None,
    build_workspace_context=lambda: "",
    tool_packs=None,
    run_once_use_tool_packs=False,
    build_system_prompt=lambda **kwargs: "system prompt",
    build_initial_user_message=lambda today, briefing_file: "Run the analyst loop.",
    describe_market_status=lambda: "closed",
    on_fallback=None,
    retry_config=None,
    state_subdir="analyst",
    state_file_name="state.json",
    format_tool_catalog=lambda *args, **kwargs: "",
    build_tool_packs_section=lambda *args, **kwargs: "",
  )
  for key, value in overrides.items():
    setattr(profile, key, value)
  return profile


def test_append_state_update_event_persists_payload(tmp_path: Path) -> None:
  log = AgentSessionLog(path=tmp_path / "sessions" / "state-update.jsonl")

  _run(
    autonomous_entry._append_state_update_event(
      log,
      payload={"alerts": ["Check filings"], "active_servers": ["fmp-mcp"]},
      runner_id="runner_test",
      model_name="claude-sonnet-4-6",
    )
  )

  entries, _ = _run(log.query(event_types={"state_update"}, order="asc"))
  assert len(entries) == 1
  assert entries[0].event["payload"] == {"alerts": ["Check filings"], "active_servers": ["fmp-mcp"]}
  assert entries[0].event["runner_id"] == "runner_test"
  assert entries[0].event["model"] == "claude-sonnet-4-6"
  assert isinstance(entries[0].event["generated_at"], float)


def test_run_once_does_not_read_or_write_state_json_and_appends_state_update(
  monkeypatch,
  tmp_path: Path,
) -> None:
  log = AgentSessionLog(path=tmp_path / "sessions" / "run-once.jsonl")
  captured: dict[str, Any] = {}
  workspace = tmp_path / "workspace"
  state_dir = workspace / "notes" / "analyst"
  state_dir.mkdir(parents=True, exist_ok=True)
  state_path = state_dir / "state.json"
  state_path.write_text("{not valid json", encoding="utf-8")

  async def _fake_build_runtime_context(*args: Any, **kwargs: Any):
    _ = args, kwargs
    return SimpleNamespace(
      workspace=workspace,
      tool_catalog="catalog",
      tool_packs_section="",
      connected_servers={"fmp-mcp", "macro-mcp"},
      active_servers={"fmp-mcp"},
    )

  def _fake_create_session_objects(*args: Any, **kwargs: Any):
    _ = args, kwargs
    captured["create_kwargs"] = kwargs
    return EventLog(), SimpleNamespace(_runner_id="runner_test")

  async def _fake_run_agent_session(*args: Any, **kwargs: Any):
    _ = args, kwargs
    return autonomous_entry.RunOutput(
      response=(
        "Run complete.\n\n"
        "## STATE_UPDATE_JSON\n"
        "```json\n"
        '{"alerts":["Check filings"],"active_servers":["fmp-mcp"]}\n'
        "```"
      ),
      tools_used=["memory_write"],
      usage={"input_tokens": 3, "output_tokens": 5},
      error=None,
      timed_out=False,
    )

  async def _fake_shutdown_session(_session_id: str, _mcp_client_manager: object = None) -> None:
    return None

  def _unexpected_read(_path: Path) -> dict[str, Any]:
    raise AssertionError("state.json should not be read in Phase 3a")

  def _unexpected_write(_path: Path, _payload: dict[str, Any]) -> None:
    raise AssertionError("state.json should not be written in Phase 3a")

  def _build_initial_user_message(today: str, briefing_file: str) -> str:
    captured["today"] = today
    captured["briefing_file"] = briefing_file
    return "Run the analyst loop."

  profile = SimpleNamespace(
    name="analyst",
    run_once_session_id_template="{profile}:{today}",
    briefing_file_template="analyst/{date}.md",
    run_once_excluded_tools=None,
    excluded_tools=set(),
    model="claude-sonnet-4-6",
    max_turns=5,
    timeout_seconds=60.0,
    per_turn_timeout=None,
    max_tokens=16000,
    client_timeout=30.0,
    max_budget_usd=2.0,
    compaction_instructions=None,
    build_workspace_context=lambda: "",
    tool_packs=None,
    run_once_use_tool_packs=False,
    build_system_prompt=lambda **kwargs: "system prompt",
    build_initial_user_message=_build_initial_user_message,
    describe_market_status=lambda: "closed",
    on_fallback=None,
    retry_config=None,
    state_subdir="analyst",
    state_file_name="state.json",
    format_tool_catalog=lambda *args, **kwargs: "",
    build_tool_packs_section=lambda *args, **kwargs: "",
  )

  monkeypatch.setattr(autonomous_entry, "build_agent_session_log", lambda **kwargs: log)
  monkeypatch.setattr(autonomous_entry, "_build_runtime_context", _fake_build_runtime_context)
  monkeypatch.setattr(autonomous_entry, "create_session_objects", _fake_create_session_objects)
  monkeypatch.setattr(autonomous_entry, "run_agent_session", _fake_run_agent_session)
  monkeypatch.setattr(autonomous_entry, "_shutdown_session", _fake_shutdown_session)
  monkeypatch.setattr(autonomous_entry, "send_telegram_summary", lambda *args, **kwargs: None)
  monkeypatch.setattr(autonomous_entry.workspace_state_io, "_safe_read_json", _unexpected_read)
  monkeypatch.setattr(autonomous_entry.workspace_state_io, "_atomic_write_json", _unexpected_write)

  exit_code = _run(autonomous_entry.run_once(profile))

  assert exit_code == 0
  assert state_path.read_text(encoding="utf-8") == "{not valid json"
  assert isinstance(captured["create_kwargs"]["operator_pause_event"], asyncio.Event)

  entries, _ = _run(log.query(event_types={"state_update"}, order="asc"))
  assert len(entries) == 1
  assert entries[0].event["payload"] == {
    "alerts": ["Check filings"],
    "active_servers": ["fmp-mcp"],
  }
  assert entries[0].event["runner_id"] == "runner_test"


def test_run_once_skips_summary_on_interrupted_run(
  monkeypatch,
  tmp_path: Path,
) -> None:
  log = AgentSessionLog(path=tmp_path / "sessions" / "run-once-interrupted.jsonl")
  summary_calls = 0
  workspace = tmp_path / "workspace"
  state_dir = workspace / "notes" / "analyst"
  state_dir.mkdir(parents=True, exist_ok=True)
  state_path = state_dir / "state.json"
  state_path.write_text("{still invalid json", encoding="utf-8")

  async def _fake_build_runtime_context(*args: Any, **kwargs: Any):
    _ = args, kwargs
    return SimpleNamespace(
      workspace=workspace,
      tool_catalog="catalog",
      tool_packs_section="",
      connected_servers={"fmp-mcp"},
      active_servers={"fmp-mcp"},
    )

  def _fake_create_session_objects(*args: Any, **kwargs: Any):
    _ = args, kwargs
    return EventLog(), SimpleNamespace(_runner_id="runner_test")

  async def _fake_run_agent_session(*args: Any, **kwargs: Any):
    _ = args, kwargs
    return autonomous_entry.RunOutput(
      response="Interrupted run.",
      tools_used=[],
      usage={},
      error="budget_exceeded",
      timed_out=True,
    )

  async def _fake_shutdown_session(_session_id: str, _mcp_client_manager: object = None) -> None:
    return None

  def _unexpected_read(_path: Path) -> dict[str, Any]:
    raise AssertionError("state.json should not be read in Phase 3a/3b")

  def _unexpected_write(_path: Path, _payload: dict[str, Any]) -> None:
    raise AssertionError("state.json should not be written in Phase 3a/3b")

  async def _fake_generate_summary(_log: AgentSessionLog) -> None:
    nonlocal summary_calls
    summary_calls += 1
    return None

  profile = SimpleNamespace(
    name="analyst",
    run_once_session_id_template="{profile}:{today}",
    briefing_file_template="analyst/{date}.md",
    run_once_excluded_tools=None,
    excluded_tools=set(),
    model="claude-sonnet-4-6",
    max_turns=5,
    timeout_seconds=60.0,
    per_turn_timeout=None,
    max_tokens=16000,
    client_timeout=30.0,
    max_budget_usd=2.0,
    compaction_instructions=None,
    build_workspace_context=lambda: "",
    tool_packs=None,
    run_once_use_tool_packs=False,
    build_system_prompt=lambda **kwargs: "system prompt",
    build_initial_user_message=lambda today, briefing_file: "Run the analyst loop.",
    describe_market_status=lambda: "closed",
    on_fallback=None,
    retry_config=None,
    state_subdir="analyst",
    state_file_name="state.json",
    format_tool_catalog=lambda *args, **kwargs: "",
    build_tool_packs_section=lambda *args, **kwargs: "",
  )

  monkeypatch.setattr(autonomous_entry, "build_agent_session_log", lambda **kwargs: log)
  monkeypatch.setattr(autonomous_entry, "_build_runtime_context", _fake_build_runtime_context)
  monkeypatch.setattr(autonomous_entry, "create_session_objects", _fake_create_session_objects)
  monkeypatch.setattr(autonomous_entry, "run_agent_session", _fake_run_agent_session)
  monkeypatch.setattr(autonomous_entry, "_shutdown_session", _fake_shutdown_session)
  monkeypatch.setattr(autonomous_entry, "send_telegram_summary", lambda *args, **kwargs: None)
  # entry.py imports generate_analyst_session_summary lazily inside
  # _analyst_context_helpers(); patch that seam (both run_once call sites use it)
  # rather than a module-level attr that does not exist.
  _real_builder, _ = autonomous_entry._analyst_context_helpers()
  monkeypatch.setattr(
    autonomous_entry,
    "_analyst_context_helpers",
    lambda: (_real_builder, _fake_generate_summary),
  )
  monkeypatch.setattr(autonomous_entry.workspace_state_io, "_safe_read_json", _unexpected_read)
  monkeypatch.setattr(autonomous_entry.workspace_state_io, "_atomic_write_json", _unexpected_write)

  exit_code = _run(autonomous_entry.run_once(profile))

  assert exit_code != 0
  assert summary_calls == 0
  assert state_path.read_text(encoding="utf-8") == "{still invalid json"
  summary_entries, _ = _run(log.query(event_types={"summary"}, order="asc"))
  state_entries, _ = _run(log.query(event_types={"state_update"}, order="asc"))
  assert summary_entries == []
  assert state_entries == []


def test_run_once_budget_exceeded_with_fresh_briefing_returns_degraded_success(
  monkeypatch,
  tmp_path: Path,
  caplog,
) -> None:
  log = AgentSessionLog(path=tmp_path / "sessions" / "run-once-budget-recovered.jsonl")
  captured: dict[str, Any] = {}
  workspace = tmp_path / "workspace"

  async def _fake_build_runtime_context(*args: Any, **kwargs: Any):
    _ = args, kwargs
    return SimpleNamespace(
      workspace=workspace,
      tool_catalog="catalog",
      tool_packs_section="",
      connected_servers={"fmp-mcp"},
      active_servers={"fmp-mcp"},
    )

  def _fake_create_session_objects(*args: Any, **kwargs: Any):
    _ = args, kwargs
    return EventLog(), SimpleNamespace(_runner_id="runner_budget")

  async def _fake_run_agent_session(*args: Any, **kwargs: Any):
    _ = args, kwargs
    return autonomous_entry.RunOutput(
      response="[Budget limit reached: $2.0226 >= $2.0000]",
      tools_used=["memory_read"],
      usage={"estimated_cost": 2.0226},
      error=None,
      timed_out=False,
      budget_exceeded=True,
    )

  async def _fake_shutdown_session(_session_id: str, _mcp_client_manager: object = None) -> None:
    return None

  def _fallback(state: dict[str, Any], today: str) -> str:
    captured["fallback_state"] = state
    return f"# Analyst Briefing - {today} (AUTO-RECOVERY)\nRecovered from budget cap.\n"

  def _fake_send_telegram_summary(_profile: Any, run_output: Any, _briefing_file: str, **kwargs: Any) -> None:
    captured["summary_output"] = run_output
    captured["summary_state"] = kwargs["state"]

  profile = _analyst_run_once_profile(on_fallback=_fallback)
  monkeypatch.setattr(autonomous_entry, "build_agent_session_log", lambda **kwargs: log)
  monkeypatch.setattr(autonomous_entry, "_build_runtime_context", _fake_build_runtime_context)
  monkeypatch.setattr(autonomous_entry, "create_session_objects", _fake_create_session_objects)
  monkeypatch.setattr(autonomous_entry, "run_agent_session", _fake_run_agent_session)
  monkeypatch.setattr(autonomous_entry, "_shutdown_session", _fake_shutdown_session)
  monkeypatch.setattr(autonomous_entry, "send_telegram_summary", _fake_send_telegram_summary)
  caplog.set_level("WARNING", logger="chat.autonomous_entry")

  exit_code = _run(autonomous_entry.run_once(profile))

  assert exit_code == 0
  briefing_files = list((workspace / "notes" / "analyst").glob("*.md"))
  assert len(briefing_files) == 1
  assert "Recovered from budget cap." in briefing_files[0].read_text(encoding="utf-8")
  assert captured["summary_output"].budget_exceeded is True
  assert captured["summary_state"]["budget_exceeded"] is True
  assert captured["fallback_state"]["budget_exceeded"] is True
  assert "treating run_once exit as degraded success" in caplog.text
  state_entries, _ = _run(log.query(event_types={"state_update"}, order="asc"))
  assert state_entries == []


def test_run_once_budget_exceeded_without_fresh_briefing_returns_budget_exit(
  monkeypatch,
  tmp_path: Path,
) -> None:
  log = AgentSessionLog(path=tmp_path / "sessions" / "run-once-budget-unrecovered.jsonl")
  workspace = tmp_path / "workspace"

  async def _fake_build_runtime_context(*args: Any, **kwargs: Any):
    _ = args, kwargs
    return SimpleNamespace(
      workspace=workspace,
      tool_catalog="catalog",
      tool_packs_section="",
      connected_servers={"fmp-mcp"},
      active_servers={"fmp-mcp"},
    )

  def _fake_create_session_objects(*args: Any, **kwargs: Any):
    _ = args, kwargs
    return EventLog(), SimpleNamespace(_runner_id="runner_budget")

  async def _fake_run_agent_session(*args: Any, **kwargs: Any):
    _ = args, kwargs
    return autonomous_entry.RunOutput(
      response="[Budget limit reached: $2.0226 >= $2.0000]",
      tools_used=[],
      usage={"estimated_cost": 2.0226},
      error=None,
      timed_out=False,
      budget_exceeded=True,
    )

  async def _fake_shutdown_session(_session_id: str, _mcp_client_manager: object = None) -> None:
    return None

  profile = _analyst_run_once_profile(on_fallback=lambda _state, _today: "")
  monkeypatch.setattr(autonomous_entry, "build_agent_session_log", lambda **kwargs: log)
  monkeypatch.setattr(autonomous_entry, "_build_runtime_context", _fake_build_runtime_context)
  monkeypatch.setattr(autonomous_entry, "create_session_objects", _fake_create_session_objects)
  monkeypatch.setattr(autonomous_entry, "run_agent_session", _fake_run_agent_session)
  monkeypatch.setattr(autonomous_entry, "_shutdown_session", _fake_shutdown_session)
  monkeypatch.setattr(autonomous_entry, "send_telegram_summary", lambda *args, **kwargs: None)

  exit_code = _run(autonomous_entry.run_once(profile))

  assert exit_code == 2
  assert not (workspace / "notes" / "analyst").exists()


def test_run_once_times_out_session_summary(
  monkeypatch,
  tmp_path: Path,
  caplog,
) -> None:
  log = AgentSessionLog(path=tmp_path / "sessions" / "run-once-summary-timeout.jsonl")
  workspace = tmp_path / "workspace"

  async def _fake_build_runtime_context(*args: Any, **kwargs: Any):
    _ = args, kwargs
    return SimpleNamespace(
      workspace=workspace,
      tool_catalog="catalog",
      tool_packs_section="",
      connected_servers={"fmp-mcp"},
      active_servers={"fmp-mcp"},
    )

  def _fake_create_session_objects(*args: Any, **kwargs: Any):
    _ = args, kwargs
    return EventLog(), SimpleNamespace(_runner_id="runner_test")

  async def _fake_run_agent_session(*args: Any, **kwargs: Any):
    _ = args, kwargs
    return autonomous_entry.RunOutput(
      response=(
        "Run complete.\n\n"
        "## STATE_UPDATE_JSON\n"
        "```json\n"
        '{"alerts":["Check filings"]}\n'
        "```"
      ),
      tools_used=["memory_write"],
      usage={},
      error=None,
      timed_out=False,
    )

  async def _fake_shutdown_session(_session_id: str, _mcp_client_manager: object = None) -> None:
    return None

  async def _hanging_generate_summary(_log: AgentSessionLog) -> None:
    await asyncio.sleep(60)

  profile = SimpleNamespace(
    name="analyst",
    run_once_session_id_template="{profile}:{today}",
    briefing_file_template="analyst/{date}.md",
    run_once_excluded_tools=None,
    excluded_tools=set(),
    model="claude-sonnet-4-6",
    max_turns=5,
    timeout_seconds=60.0,
    per_turn_timeout=None,
    max_tokens=16000,
    client_timeout=30.0,
    max_budget_usd=2.0,
    compaction_instructions=None,
    build_workspace_context=lambda: "",
    tool_packs=None,
    run_once_use_tool_packs=False,
    build_system_prompt=lambda **kwargs: "system prompt",
    build_initial_user_message=lambda today, briefing_file: "Run the analyst loop.",
    describe_market_status=lambda: "closed",
    on_fallback=None,
    retry_config=None,
    state_subdir="analyst",
    state_file_name="state.json",
    format_tool_catalog=lambda *args, **kwargs: "",
    build_tool_packs_section=lambda *args, **kwargs: "",
  )

  monkeypatch.setenv("ANALYST_SESSION_SUMMARY_TIMEOUT_SECONDS", "0.01")
  monkeypatch.setattr(autonomous_entry, "build_agent_session_log", lambda **kwargs: log)
  monkeypatch.setattr(autonomous_entry, "_build_runtime_context", _fake_build_runtime_context)
  monkeypatch.setattr(autonomous_entry, "create_session_objects", _fake_create_session_objects)
  monkeypatch.setattr(autonomous_entry, "run_agent_session", _fake_run_agent_session)
  monkeypatch.setattr(autonomous_entry, "_shutdown_session", _fake_shutdown_session)
  monkeypatch.setattr(autonomous_entry, "send_telegram_summary", lambda *args, **kwargs: None)
  # See note in test_run_once_skips_summary_on_interrupted_run: patch the lazy
  # _analyst_context_helpers() seam, not a nonexistent module-level attr.
  _real_builder, _ = autonomous_entry._analyst_context_helpers()
  monkeypatch.setattr(
    autonomous_entry,
    "_analyst_context_helpers",
    lambda: (_real_builder, _hanging_generate_summary),
  )
  caplog.set_level("WARNING", logger="chat.autonomous_entry")

  exit_code = _run(autonomous_entry.run_once(profile))

  assert exit_code == 0
  assert "Session summary timed out" in caplog.text
  state_entries, _ = _run(log.query(event_types={"state_update"}, order="asc"))
  summary_entries, _ = _run(log.query(event_types={"summary"}, order="asc"))
  assert len(state_entries) == 1
  assert summary_entries == []


def test_run_output_allows_state_update_rejects_interrupted_outputs() -> None:
  assert autonomous_entry._run_output_allows_state_update(
    autonomous_entry.RunOutput("done", [], {}, None, False)
  ) is True
  assert autonomous_entry._run_output_allows_state_update(
    autonomous_entry.RunOutput("paused", [], {}, None, False, operator_paused=True)
  ) is False
  assert autonomous_entry._run_output_allows_state_update(
    autonomous_entry.RunOutput("budget", [], {}, None, False, budget_exceeded=True)
  ) is False
  assert autonomous_entry._run_output_allows_state_update(
    autonomous_entry.RunOutput("max", [], {}, None, False, max_turns_reached=True)
  ) is False
  assert autonomous_entry._run_output_allows_state_update(
    autonomous_entry.RunOutput("max tokens", [], {}, None, False, max_tokens_reached=True)
  ) is False
