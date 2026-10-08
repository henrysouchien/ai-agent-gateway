"""Init-time session resume: `POST /chat/init` with `resume_session_id`.

The server resolves the caller, asks the configured resume resolver for the
caller's own log, refuses with 409 `session_resume_refused`, or binds the new
session to that log so the next turn replays it.
"""

import json
import sys
import asyncio
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.testclient import TestClient

TESTS_DIR = Path(__file__).resolve().parent
if str(TESTS_DIR) not in sys.path:
  sys.path.insert(0, str(TESTS_DIR))

from agent_gateway import (  # noqa: E402
  AgentRunner,
  AgentSessionLog,
  SessionContextBuilder,
  ToolDispatcher,
)
from agent_gateway.event_log import EventLog  # noqa: E402
from agent_gateway.mcp_client import McpClientManager  # noqa: E402
from agent_gateway.server import (  # noqa: E402
  ChatRuntime,
  GatewayServerConfig,
  create_gateway_app,
)
from agent_gateway.session import (  # noqa: E402
  AuthManager,
  SessionStream,
  session_owner_user_id,
)
from model_authority.capabilities import CAPABILITY_IDS  # noqa: E402
from test_server_multi_user import (  # noqa: E402
  _ReplayProvider,
  _consume_chat_stream,
  _resolver_result_for,
  _strict_session_driver_config,
)


API_KEY = "gateway-key"


class _ResumeResolver:
  """Answers from a table the test fills; records the caller it was given."""

  def __init__(self) -> None:
    self.outcomes: dict[str, Any] = {}
    self.calls: list[dict[str, Any]] = []

  def __call__(
    self,
    *,
    resume_session_id: str,
    user_id: str,
    channel: str | None,
    api_key_hash: str,
  ):
    self.calls.append(
      {
        "resume_session_id": resume_session_id,
        "user_id": user_id,
        "channel": channel,
        "api_key_hash": api_key_hash,
      }
    )
    return self.outcomes.get(resume_session_id, "not_found")


def _log_path(logs_dir: Path, session) -> Path:
  return logs_dir / f"agentsess_{session.session_id}_{session.user_id}.jsonl"


def _make_app(
  logs_dir: Path,
  provider: _ReplayProvider,
  resume_resolver: _ResumeResolver | None,
) -> FastAPI:
  async def _credentials_resolver(_api_key: str, init_request):
    return _resolver_result_for(init_request.user_id, channel="excel")

  async def _build_chat_runtime(
    session,
    request,
    channel,
    auth_manager,
    *,
    storage_root: Path | None = None,
  ):
    _ = channel, auth_manager, storage_root
    capability_execution = request.capability_execution
    assert capability_execution is not None
    # The product factory's rule: a resumed session opens the bound log,
    # any other session mints one keyed by its own id; a fork also reads its
    # source log, without binding it, to seed that new log.
    session_log = session.agent_session_log or AgentSessionLog(
      path=session.resume_log_path or _log_path(logs_dir, session),
      gateway_session=session,
    )
    session.agent_session_log = session_log
    fork_source_log = (
      AgentSessionLog(
        path=session.fork_source_log_path,
        _create_active_on_open=False,
        _repair_manifest_on_open=False,
      )
      if session.fork_source_log_path is not None
      else None
    )

    def _build_runner(event_log, session_id, _started_at=None):
      return AgentRunner(
        event_log=event_log,
        dispatcher=ToolDispatcher(
          mcp_client=McpClientManager(config_path=None),
          local_tool_handlers={},
          event_log=event_log,
          session_id=session_id,
        ),
        session_id=session_id,
        capability_execution=capability_execution,
        get_tool_definitions=lambda: [],
        user_id=session_owner_user_id(session),
        channel=session.channel,
        gateway_session=session,
        billing_mode="byok",
        rate_table_version="unknown",
        agent_session_log=session_log,
        context_builder=SessionContextBuilder(
          agent_session_log=session_log,
          fork_source_log=fork_source_log,
        ),
      )

    return ChatRuntime(
      system_prompt="system",
      build_runner=_build_runner,
      capability_execution=capability_execution,
    )

  def _resolve_adapter(adapter: str):
    assert adapter == "anthropic.messages"
    return provider

  return create_gateway_app(
    GatewayServerConfig(
      **_strict_session_driver_config(
        provider="anthropic",
        model="stub-model",
        effort="none",
        tenant_id="test-product",
        auth_config={
          "provider": "anthropic",
          "billing_mode": "byok",
          "rate_table_version": "unknown",
          "api_key": "k",
        },
      ),
      default_provider=provider,
      capability_adapter_resolver=_resolve_adapter,
      credentials_resolver=_credentials_resolver,
      build_chat_runtime=_build_chat_runtime,
      session_resume_resolver=resume_resolver,
    )
  )


def _post_init(
  client: TestClient,
  *,
  user_id: str = "alice",
  api_key: str = API_KEY,
  resume_session_id: str | None = None,
  fork_from_session_id: str | None = None,
):
  payload: dict[str, Any] = {"api_key": api_key, "user_id": user_id}
  if resume_session_id is not None:
    payload["resume_session_id"] = resume_session_id
  if fork_from_session_id is not None:
    payload["fork_from_session_id"] = fork_from_session_id
  return client.post("/api/chat/init", json=payload)


def _ready_session(client: TestClient, init: dict[str, Any]) -> dict[str, Any]:
  app = client.app
  assert isinstance(app, FastAPI)
  session = app.state.auth.session_store.get_session(init["session_id"])
  assert session is not None
  session.model_entitled_capabilities = frozenset(CAPABILITY_IDS)
  session.model_entitled_keys = frozenset(
    app.state.gateway_config.model_registry.models
  )
  return {**init, "session_token": app.state.auth.issue_token(session)}


def _turn(client: TestClient, init: dict[str, Any], text: str, user_id: str = "alice") -> None:
  _consume_chat_stream(
    client,
    init["session_token"],
    {"user_id": user_id, "messages": [{"role": "user", "content": text}]},
  )


def _events(path: Path) -> list[dict[str, Any]]:
  return [
    json.loads(line)["event"]
    for line in path.read_text(encoding="utf-8").splitlines()
    if line.strip()
  ]


def _tree(root: Path) -> dict[str, tuple[int, int]]:
  return {
    str(path.relative_to(root)): (path.stat().st_size, path.stat().st_mtime_ns)
    for path in sorted(root.rglob("*"))
    if path.is_file()
  }


def _first_session(tmp_path: Path, resolver: _ResumeResolver) -> tuple[Path, str]:
  """Run one turn as alice on a gateway that then goes away (a restart)."""

  logs_dir = tmp_path / "logs"
  provider = _ReplayProvider(["The codeword is PELICAN."])
  with TestClient(_make_app(logs_dir, provider, resolver)) as client:
    init = _ready_session(client, _post_init(client).json())
    _turn(client, init, "Remember the codeword PELICAN.")
  log_path = logs_dir / f"agentsess_{init['session_id']}_alice.jsonl"
  assert log_path.is_file()
  return log_path, init["session_id"]


def test_resume_after_restart_replays_prior_turns_and_appends_to_prior_log(
  tmp_path: Path,
) -> None:
  resolver = _ResumeResolver()
  log_path, first_id = _first_session(tmp_path, resolver)
  resolver.outcomes[first_id] = log_path
  logs_dir = log_path.parent

  provider = _ReplayProvider(["You told me PELICAN."])
  app = _make_app(logs_dir, provider, resolver)
  with TestClient(app) as client:
    response = _post_init(client, resume_session_id=first_id)
    assert response.status_code == 200, response.text
    body = response.json()
    resumed = _ready_session(client, body)
    _turn(client, resumed, "What was the codeword?")

  assert body["resumed_from_session_id"] == first_id
  assert body["session_id"] != first_id
  assert resolver.calls == [
    {
      "resume_session_id": first_id,
      "user_id": "alice",
      "channel": "excel",
      "api_key_hash": AuthManager.hash_api_key(API_KEY),
    }
  ]
  model_input = json.dumps(provider.captured_messages[-1], default=str)
  assert "Remember the codeword PELICAN." in model_input
  assert "The codeword is PELICAN." in model_input
  assert "What was the codeword?" in model_input
  # The turn went to the prior log; no log was minted for the new id.
  assert sorted(p.name for p in logs_dir.glob("*.jsonl")) == [log_path.name]
  attaches = [event for event in _events(log_path) if event["type"] == "attach"]
  assert [event["gateway_session_id"] for event in attaches] == [
    first_id,
    body["session_id"],
  ]
  assert {
    (event["user_id"], event["channel"], event["api_key_hash"])
    for event in attaches
  } == {("alice", "excel", AuthManager.hash_api_key(API_KEY))}
  assert any(
    event["type"] == "user_message" and event["content"] == "What was the codeword?"
    for event in _events(log_path)
  )


def test_chained_resume_presenting_the_original_id_appends_to_the_same_log(
  tmp_path: Path,
) -> None:
  resolver = _ResumeResolver()
  log_path, first_id = _first_session(tmp_path, resolver)
  resolver.outcomes[first_id] = log_path
  logs_dir = log_path.parent

  with TestClient(_make_app(logs_dir, _ReplayProvider(["Second."]), resolver)) as client:
    second = _post_init(client, resume_session_id=first_id)
    assert second.status_code == 200, second.text
    _turn(client, _ready_session(client, second.json()), "Second turn.")

  provider = _ReplayProvider(["Third."])
  with TestClient(_make_app(logs_dir, provider, resolver)) as client:
    third = _post_init(client, resume_session_id=first_id)
    assert third.status_code == 200, third.text
    _turn(client, _ready_session(client, third.json()), "Third turn.")

  assert third.json()["resumed_from_session_id"] == first_id
  model_input = json.dumps(provider.captured_messages[-1], default=str)
  assert "Remember the codeword PELICAN." in model_input
  assert "Second turn." in model_input
  assert sorted(p.name for p in logs_dir.glob("*.jsonl")) == [log_path.name]
  assert [
    event["gateway_session_id"]
    for event in _events(log_path)
    if event["type"] == "attach"
  ] == [first_id, second.json()["session_id"], third.json()["session_id"]]


def test_refusal_is_409_with_reason_and_registers_no_session_or_file(
  tmp_path: Path,
) -> None:
  resolver = _ResumeResolver()
  log_path, _first_id = _first_session(tmp_path, resolver)
  resolver.outcomes.update(
    {
      "sess_ambiguous": "ambiguous",
      "sess_unrecorded": "no_ownership_record",
    }
  )
  before = _tree(tmp_path)

  app = _make_app(log_path.parent, _ReplayProvider([]), resolver)
  with TestClient(app) as client:
    responses = {
      presented: _post_init(client, resume_session_id=presented)
      for presented in ("sess_missing", "sess_ambiguous", "sess_unrecorded")
    }
    live_sessions = app.state.auth.session_store.visible_sessions_snapshot()

  assert {
    presented: (response.status_code, response.json())
    for presented, response in responses.items()
  } == {
    "sess_missing": (
      409,
      {"error": "session_resume_refused", "reason": "not_found"},
    ),
    "sess_ambiguous": (
      409,
      {"error": "session_resume_refused", "reason": "ambiguous"},
    ),
    "sess_unrecorded": (
      409,
      {"error": "session_resume_refused", "reason": "no_ownership_record"},
    ),
  }
  assert live_sessions == ()
  assert _tree(tmp_path) == before


def test_resume_without_a_configured_resolver_reads_not_found(tmp_path: Path) -> None:
  app = _make_app(tmp_path / "logs", _ReplayProvider([]), None)
  with TestClient(app) as client:
    response = _post_init(client, resume_session_id="sess_anything")
    live_sessions = app.state.auth.session_store.visible_sessions_snapshot()

  assert response.status_code == 409
  assert response.json() == {
    "error": "session_resume_refused",
    "reason": "not_found",
  }
  assert live_sessions == ()


def test_resume_refuses_in_use_while_the_prior_session_is_answering(
  tmp_path: Path,
) -> None:
  resolver = _ResumeResolver()
  logs_dir = tmp_path / "logs"
  app = _make_app(logs_dir, _ReplayProvider(["First."]), resolver)
  with TestClient(app) as client:
    prior = _ready_session(client, _post_init(client).json())
    _turn(client, prior, "First turn.")
    store = app.state.auth.session_store
    prior_session = store.get_session(prior["session_id"])
    resolver.outcomes[prior["session_id"]] = _log_path(logs_dir, prior_session)

    async def _answering() -> None:
      await asyncio.Event().wait()

    async def _start_turn() -> SessionStream:
      task = asyncio.get_running_loop().create_task(_answering())
      await asyncio.sleep(0)
      return SessionStream(event_log=EventLog(), runner_task=task)

    portal = client.portal
    assert portal is not None
    prior_session.active_turn = portal.call(_start_turn)
    before = _tree(tmp_path)
    response = _post_init(client, resume_session_id=prior["session_id"])
    still_live = store.get_session(prior["session_id"])
    live_ids = {session.session_id for session in store.visible_sessions_snapshot()}
    portal.call(_cancel_turn, prior_session.active_turn)

  assert response.status_code == 409
  assert response.json() == {"error": "session_resume_refused", "reason": "in_use"}
  assert still_live is prior_session
  assert live_ids == {prior["session_id"]}
  assert _tree(tmp_path) == before


async def _cancel_turn(turn: SessionStream) -> None:
  assert turn.runner_task is not None
  turn.runner_task.cancel()


def test_resume_retires_the_callers_idle_prior_session_on_the_same_log(
  tmp_path: Path,
) -> None:
  resolver = _ResumeResolver()
  logs_dir = tmp_path / "logs"
  provider = _ReplayProvider(["First.", "Second."])
  app = _make_app(logs_dir, provider, resolver)
  with TestClient(app) as client:
    prior = _ready_session(client, _post_init(client).json())
    _turn(client, prior, "First turn.")
    store = app.state.auth.session_store
    log_path = _log_path(logs_dir, store.get_session(prior["session_id"]))
    resolver.outcomes[prior["session_id"]] = log_path

    response = _post_init(client, resume_session_id=prior["session_id"])
    assert response.status_code == 200, response.text
    resumed = _ready_session(client, response.json())
    live_ids = {session.session_id for session in store.visible_sessions_snapshot()}
    _turn(client, resumed, "Second turn.")

  assert store.get_session(prior["session_id"]) is None
  assert live_ids == {resumed["session_id"]}
  assert "First turn." in json.dumps(provider.captured_messages[-1], default=str)
  assert sorted(p.name for p in logs_dir.glob("*.jsonl")) == [log_path.name]


def _answering_turn(client: TestClient) -> SessionStream:
  async def _answering() -> None:
    await asyncio.Event().wait()

  async def _start_turn() -> SessionStream:
    task = asyncio.get_running_loop().create_task(_answering())
    await asyncio.sleep(0)
    return SessionStream(event_log=EventLog(), runner_task=task)

  portal = client.portal
  assert portal is not None
  return portal.call(_start_turn)


def test_fork_keeps_the_idle_source_session_and_replays_its_history(
  tmp_path: Path,
) -> None:
  resolver = _ResumeResolver()
  logs_dir = tmp_path / "logs"
  provider = _ReplayProvider(["The codeword is PELICAN.", "PELICAN.", "Back."])
  app = _make_app(logs_dir, provider, resolver)
  with TestClient(app) as client:
    original = _ready_session(client, _post_init(client).json())
    _turn(client, original, "Remember the codeword PELICAN.")
    store = app.state.auth.session_store
    original_session = store.get_session(original["session_id"])
    source_log = _log_path(logs_dir, original_session)
    resolver.outcomes[original["session_id"]] = source_log
    source_before = source_log.read_bytes()

    response = _post_init(client, fork_from_session_id=original["session_id"])
    assert response.status_code == 200, response.text
    body = response.json()
    fork_session = store.get_session(body["session_id"])
    fork_state = (fork_session.fork_source_log_path, fork_session.resume_log_path)
    live_after_fork = {session.session_id for session in store.visible_sessions_snapshot()}
    fork = _ready_session(client, body)
    _turn(client, fork, "What was the codeword?")
    fork_model_input = json.dumps(provider.captured_messages[-1], default=str)
    source_after_fork = source_log.read_bytes()

    # The original workbook still resumes its own log while the fork lives.
    resumed_response = _post_init(client, resume_session_id=original["session_id"])
    assert resumed_response.status_code == 200, resumed_response.text
    resumed = _ready_session(client, resumed_response.json())
    _turn(client, resumed, "Are we still on the original?")
    original_model_input = json.dumps(provider.captured_messages[-1], default=str)
    live_after_resume = {session.session_id for session in store.visible_sessions_snapshot()}

  assert body["forked_from_session_id"] == original["session_id"]
  assert body["resumed_from_session_id"] is None
  assert body["session_id"] != original["session_id"]
  assert resolver.calls[0]["resume_session_id"] == original["session_id"]
  assert fork_state == (source_log, None)
  assert live_after_fork == {original["session_id"], body["session_id"]}
  assert "Remember the codeword PELICAN." in fork_model_input
  assert "The codeword is PELICAN." in fork_model_input
  assert source_after_fork == source_before
  fork_log = _log_path(logs_dir, fork_session)
  fork_events = _events(fork_log)
  assert [event["type"] for event in fork_events].count("fork_seed") == 1
  assert any(
    event["type"] == "user_message" and event["content"] == "What was the codeword?"
    for event in fork_events
  )
  assert "What was the codeword?" not in original_model_input
  assert "Remember the codeword PELICAN." in original_model_input
  assert live_after_resume == {body["session_id"], resumed["session_id"]}
  assert sorted(p.name for p in logs_dir.glob("*.jsonl")) == sorted(
    [source_log.name, fork_log.name]
  )


def test_fork_is_not_refused_while_the_source_session_is_answering(
  tmp_path: Path,
) -> None:
  resolver = _ResumeResolver()
  logs_dir = tmp_path / "logs"
  app = _make_app(logs_dir, _ReplayProvider(["First."]), resolver)
  with TestClient(app) as client:
    original = _ready_session(client, _post_init(client).json())
    _turn(client, original, "First turn.")
    store = app.state.auth.session_store
    original_session = store.get_session(original["session_id"])
    resolver.outcomes[original["session_id"]] = _log_path(logs_dir, original_session)
    original_session.active_turn = _answering_turn(client)

    response = _post_init(client, fork_from_session_id=original["session_id"])
    still_live = store.get_session(original["session_id"])
    fork_session = store.get_session(response.json().get("session_id", ""))
    live_ids = {session.session_id for session in store.visible_sessions_snapshot()}
    portal = client.portal
    assert portal is not None
    portal.call(_cancel_turn, original_session.active_turn)

  assert response.status_code == 200, response.text
  assert response.json()["forked_from_session_id"] == original["session_id"]
  assert still_live is original_session
  assert fork_session is not None
  assert fork_session.fork_source_log_path == _log_path(logs_dir, original_session)
  assert fork_session.resume_log_path is None
  assert live_ids == {original["session_id"], fork_session.session_id}


def test_fork_refusal_is_the_resume_refusal_and_both_ids_are_rejected(
  tmp_path: Path,
) -> None:
  resolver = _ResumeResolver()
  resolver.outcomes["sess_ambiguous"] = "ambiguous"
  app = _make_app(tmp_path / "logs", _ReplayProvider([]), resolver)
  with TestClient(app) as client:
    refused = {
      presented: _post_init(client, fork_from_session_id=presented)
      for presented in ("sess_missing", "sess_ambiguous")
    }
    both = _post_init(
      client,
      resume_session_id="sess_missing",
      fork_from_session_id="sess_missing",
    )
    live_sessions = app.state.auth.session_store.visible_sessions_snapshot()

  assert {
    presented: (response.status_code, response.json())
    for presented, response in refused.items()
  } == {
    "sess_missing": (
      409,
      {"error": "session_resume_refused", "reason": "not_found"},
    ),
    "sess_ambiguous": (
      409,
      {"error": "session_resume_refused", "reason": "ambiguous"},
    ),
  }
  assert both.status_code == 422
  assert "mutually exclusive" in both.text
  assert [call["resume_session_id"] for call in resolver.calls] == [
    "sess_missing",
    "sess_ambiguous",
  ]
  assert live_sessions == ()
