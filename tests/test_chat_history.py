"""Bearer-bound, display-only pages over the existing interactive log."""

import json
from pathlib import Path
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


from agent_gateway import AgentSessionLog, EventLog  # noqa: E402
from agent_gateway.server import ChatTurnInputs  # noqa: E402
from agent_gateway.server_chat_helpers import (  # noqa: E402
  _dispatch_chat_turn,
  prepare_session_driver_turn,
)
from agent_gateway.server_models import ChatMessage  # noqa: E402
from agent_gateway.session import GatewaySession, SessionStream  # noqa: E402
from test_session_resume import (  # noqa: E402
  _ResumeResolver,
  _first_session,
  _make_app,
  _post_init,
  _ready_session,
  _tree,
  _turn,
)
from test_server_multi_user import _ReplayProvider  # noqa: E402
from test_server_multi_user import _consume_chat_stream  # noqa: E402


def _session(client: TestClient, init: dict[str, Any]) -> GatewaySession:
  app = client.app
  assert isinstance(app, FastAPI)
  session = app.state.auth.session_store.get_session(init["session_id"])
  assert session is not None
  return session


def _history(client: TestClient, init: dict[str, Any], **params: Any):
  return client.get(
    "/api/chat/history",
    headers={"Authorization": f"Bearer {init['session_token']}"},
    params=params,
  )


def _bind_log(client: TestClient, init: dict[str, Any], path: Path) -> AgentSessionLog:
  session = _session(client, init)
  log = AgentSessionLog(path=path, gateway_session=session)
  session.agent_session_log = log
  return log


def test_fresh_history_is_empty_private_and_does_not_create_a_log(tmp_path: Path) -> None:
  with TestClient(_make_app(tmp_path / "logs", _ReplayProvider([]), None)) as client:
    init = _post_init(client).json()
    before = _tree(tmp_path)
    response = _history(client, init)

    assert response.status_code == 200, response.text
    assert response.headers["cache-control"] == "private, no-store"
    assert response.json() == {
      "schema_version": 1,
      "entries": [],
      "next_before_seq": None,
    }
    assert _session(client, init).agent_session_log is None
    assert _tree(tmp_path) == before


def test_completed_turns_return_durable_sequence_and_timestamp(tmp_path: Path) -> None:
  provider = _ReplayProvider(["First answer.", "Second answer."])
  with TestClient(_make_app(tmp_path / "logs", provider, None)) as client:
    init = _ready_session(client, _post_init(client).json())
    _turn(client, init, "First question.")
    _turn(client, init, "Second question.")
    log = _session(client, init).agent_session_log
    assert log is not None
    durable = [
      row
      for line in log.path.read_text(encoding="utf-8").splitlines()
      if (row := json.loads(line))["event"]["type"] in {"user_message", "assistant_message"}
    ]
    response = _history(client, init)

    assert response.status_code == 200, response.text
    entries = response.json()["entries"]
    assert [(entry["role"], entry["content"]) for entry in entries] == [
      ("user", "First question."),
      ("assistant", "First answer."),
      ("user", "Second question."),
      ("assistant", "Second answer."),
    ]
    assert [(entry["seq"], entry["timestamp"]) for entry in entries] == [
      (row["seq"], row["timestamp"]) for row in durable
    ]
    assert response.json()["next_before_seq"] is None


@pytest.mark.parametrize("as_blocks", [False, True])
def test_injected_context_stays_in_model_replay_but_not_reloaded_history(
  tmp_path: Path,
  as_blocks: bool,
) -> None:
  context = (
    '## Current Workbook Context\n```json\n'
    '{"channel": "excel", "workbook": "earnings.xlsx", "selection": "B5"}\n```'
  )
  text = "What drove revenue?"
  analyst_content = [{"type": "text", "text": text}] if as_blocks else text
  model_content = (
    [*analyst_content, {"type": "text", "text": context}]
    if isinstance(analyst_content, list)
    else f"{text}\n\n{context}"
  )
  resolver = _ResumeResolver()
  provider = _ReplayProvider(["Revenue grew.", "Still grew."])
  app = _make_app(
    tmp_path / "logs", provider, resolver, turn_context_message=context,
  )
  with TestClient(app) as client:
    init = _ready_session(client, _post_init(client).json())
    if as_blocks:
      # HTTP chat accepts text; exercise the existing internal block-list
      # injection path without widening that request contract.
      session = _session(client, init)
      builder = app.state.gateway_build_chat_runtime
      inputs = ChatTurnInputs(
        messages=[ChatMessage.model_construct(role="user", content=analyst_content)],
        request_id="block-provenance",
        context={"channel": "excel", "workbook": "earnings.xlsx"},
        metadata={},
        model_key=None,
      )
      prepared = prepare_session_driver_turn(session, inputs, build_chat_runtime=builder)

      async def internal_turn():
        async def on_event(_event: dict[str, Any]) -> None:
          pass

        return await _dispatch_chat_turn(
          session, inputs,
          turn=SessionStream(
            event_log=EventLog(session_id=session.session_id), runner_task=None,
          ),
          on_event=on_event,
          build_chat_runtime=builder,
          transcript_dir=None,
          prepared_turn=prepared,
        )

      portal = client.portal
      assert portal is not None
      assert portal.call(internal_turn).state == "completed"
    else:
      _consume_chat_stream(
        client,
        init["session_token"],
        {
          "user_id": "alice",
          "messages": [{"role": "user", "content": analyst_content}],
          "context": {"channel": "excel", "workbook": "earnings.xlsx"},
        },
      )
    log = _session(client, init).agent_session_log
    assert log is not None
    rows = [json.loads(line) for line in log.path.read_text(encoding="utf-8").splitlines()]
    user = next(row["event"] for row in rows if row["event"]["type"] == "user_message")
    assert user["content"] == model_content
    assert user["analyst_content"] == analyst_content
    assert provider.captured_messages[0][-1] == {
      "role": "user", "content": model_content,
    }
    first_history = _history(client, init)
    assert first_history.status_code == 200, first_history.text
    assert [entry["content"] for entry in first_history.json()["entries"]] == [
      text, "Revenue grew.",
    ]

  resolver.outcomes[init["session_id"]] = log.path
  with TestClient(_make_app(log.path.parent, provider, resolver)) as client:
    resumed = _ready_session(
      client, _post_init(client, resume_session_id=init["session_id"]).json(),
    )
    assert _history(client, resumed).json() == first_history.json()
    _turn(client, resumed, "Explain further.")
    assert provider.captured_messages[-1][0] == {
      "role": "user", "content": model_content,
    }
    assert all(
      "analyst_content" not in message
      for messages in provider.captured_messages
      for message in messages
    )


def test_legacy_user_record_refuses_history_without_guessing_analyst_text(tmp_path: Path) -> None:
  with TestClient(_make_app(tmp_path / "logs", _ReplayProvider([]), None)) as client:
    init = _post_init(client).json()
    log = _bind_log(client, init, tmp_path / "logs" / "legacy.jsonl")
    legacy = log.append_sync({
      "type": "user_message", "role": "writer", "client_kind": "excel",
      "content": "Question\n\n## Current Workbook Context\nPRIVATE_CONTEXT",
    })
    response = _history(client, init)
    assert response.status_code == 409, response.text
    assert response.json() == {
      "detail": {"code": "history_analyst_content_unavailable", "seq": legacy.seq},
    }
    assert "PRIVATE_CONTEXT" not in response.text


@pytest.mark.parametrize("analyst_content", [None, True, 12, {}])
def test_history_does_not_fall_back_from_malformed_present_provenance(
  tmp_path: Path,
  analyst_content: Any,
) -> None:
  with TestClient(_make_app(tmp_path / "logs", _ReplayProvider([]), None)) as client:
    init = _post_init(client).json()
    log = _bind_log(client, init, tmp_path / "logs" / "malformed.jsonl")
    log.append_sync({
      "type": "user_message", "role": "writer", "client_kind": "excel",
      "content": "PRIVATE_MODEL_CONTEXT", "analyst_content": analyst_content,
    })
    with pytest.raises(TypeError, match="analyst content"):
      _history(client, init)


def test_restart_resume_reads_history_before_runtime_and_keeps_it_after_next_turn(
  tmp_path: Path,
) -> None:
  resolver = _ResumeResolver()
  path, original_id = _first_session(tmp_path, resolver)
  resolver.outcomes[original_id] = path
  with TestClient(_make_app(path.parent, _ReplayProvider(["Still PELICAN."]), resolver)) as client:
    init_response = _post_init(client, resume_session_id=original_id)
    assert init_response.status_code == 200, init_response.text
    init = _ready_session(client, init_response.json())
    assert init["session_id"] != original_id
    before = _tree(tmp_path)
    response = _history(client, init)

    assert response.status_code == 200, response.text
    previous_entries = response.json()["entries"]
    assert [entry["content"] for entry in previous_entries] == [
      "Remember the codeword PELICAN.", "The codeword is PELICAN.",
    ]
    assert _session(client, init).agent_session_log is None
    assert _tree(tmp_path) == before

    _turn(client, init, "What was the codeword?")
    after_turn = _history(client, init).json()
    assert after_turn["entries"][:2] == previous_entries
    assert [entry["content"] for entry in after_turn["entries"][2:]] == [
      "What was the codeword?", "Still PELICAN.",
    ]
    resumed_log = _session(client, init).agent_session_log
    assert resumed_log is not None
    assert resumed_log.path == path


@pytest.mark.parametrize("binding", ["resume", "fork"])
def test_bound_rotated_history_read_does_not_create_active_or_repair_sidecars(
  tmp_path: Path,
  monkeypatch: pytest.MonkeyPatch,
  binding: str,
) -> None:
  monkeypatch.setenv("AGENT_SESSION_LOG_MAX_ACTIVE_BYTES", "1")
  resolver = _ResumeResolver()
  path, original_id = _first_session(tmp_path, resolver)
  log = AgentSessionLog(path=path)
  log._rotate_active_if_needed_locked()
  manifest = json.loads(log.manifest_path.read_text(encoding="utf-8"))
  segment_path = log.segments_dir / manifest["segments"][0]["path"]
  path.unlink()
  path.with_suffix(".meta.json").unlink(missing_ok=True)
  segment_path.with_suffix(".meta.json").unlink()
  resolver.outcomes[original_id] = path

  with TestClient(_make_app(path.parent, _ReplayProvider([]), resolver)) as client:
    kwargs = (
      {"resume_session_id": original_id}
      if binding == "resume"
      else {"fork_from_session_id": original_id}
    )
    init_response = _post_init(client, **kwargs)
    assert init_response.status_code == 200, init_response.text
    init = init_response.json()
    before = _tree(tmp_path)
    response = _history(client, init)

    assert response.status_code == 200, response.text
    assert [entry["content"] for entry in response.json()["entries"]] == [
      "Remember the codeword PELICAN.", "The codeword is PELICAN.",
    ]
    assert _session(client, init).agent_session_log is None
    assert _tree(tmp_path) == before
    assert not path.exists()
    assert not segment_path.with_suffix(".meta.json").exists()


def test_fork_live_and_resumed_reads_never_replay_model_context_seed(tmp_path: Path) -> None:
  resolver = _ResumeResolver()
  source_path, original_id = _first_session(tmp_path, resolver)
  resolver.outcomes[original_id] = source_path
  with TestClient(_make_app(source_path.parent, _ReplayProvider(["Fork answer."]), resolver)) as client:
    response = _post_init(client, fork_from_session_id=original_id)
    assert response.status_code == 200, response.text
    fork = _ready_session(client, response.json())
    _turn(client, fork, "Fork question.")
    fork_log = _session(client, fork).agent_session_log
    assert fork_log is not None
    # The copied seed is model context, not a second display transcript.
    assert any(
      json.loads(line)["event"]["type"] == "fork_seed"
      for line in fork_log.path.read_text(encoding="utf-8").splitlines()
    )
    live_history = _history(client, fork).json()
    assert [entry["content"] for entry in live_history["entries"]] == [
      "Fork question.", "Fork answer.",
    ]

  resolver.outcomes[fork["session_id"]] = fork_log.path
  with TestClient(_make_app(source_path.parent, _ReplayProvider([]), resolver)) as client:
    response = _post_init(client, resume_session_id=fork["session_id"])
    assert response.status_code == 200, response.text
    resumed = response.json()
    before = _tree(tmp_path)
    assert _history(client, resumed).json() == live_history
    assert _tree(tmp_path) == before


def test_history_projects_only_root_analyst_and_assistant_text(tmp_path: Path) -> None:
  with TestClient(_make_app(tmp_path / "logs", _ReplayProvider([]), None)) as client:
    init = _post_init(client).json()
    log = _bind_log(client, init, tmp_path / "logs" / "display.jsonl")
    user = log.append_sync({
      "type": "user_message", "role": "writer", "client_kind": "excel",
      "content": [
        {"type": "text", "text": "Show "},
        {"type": "image", "source": {"data": "PRIVATE_IMAGE"}},
        {"type": "text", "text": "earnings"},
      ],
      "analyst_content": [
        {"type": "text", "text": "Show "},
        {"type": "image", "source": {"data": "PRIVATE_IMAGE"}},
        {"type": "text", "text": "earnings"},
      ],
      "selected_content": [{"private": "PRIVATE_SELECTION"}],
    })
    assistant = log.append_sync({
      "type": "assistant_message", "role": "writer",
      "content_blocks": [
        {"type": "thinking", "thinking": "PRIVATE_THINKING"},
        {"type": "text", "text": "Revenue "},
        {"type": "tool_use", "name": "execute", "input": {"code": "PRIVATE_ACTION"}},
        {"type": "text", "text": "grew."},
        {"type": "redacted_thinking", "data": "PRIVATE_REDACTED"},
      ],
      "workflow_output_attachments": [{"private": "PRIVATE_ATTACHMENT"}],
    })
    next_assistant = log.append_sync({
      "type": "assistant_message", "role": "writer",
      "content_blocks": [{"type": "text", "text": "More detail."}],
    })
    for event in [
      {"type": "user_message", "role": "writer", "client_kind": "operator", "content": "PRIVATE_OPERATOR", "analyst_content": "PRIVATE_OPERATOR"},
      {"type": "user_message", "role": "sub_agent", "sub_agent_id": "child", "content": "PRIVATE_CHILD_TASK", "analyst_content": "PRIVATE_CHILD_TASK"},
      {"type": "assistant_message", "role": "sub_agent", "sub_agent_id": "child", "content_blocks": [{"type": "text", "text": "PRIVATE_CHILD_RESPONSE"}]},
      {"type": "assistant_message", "role": "writer", "sub_agent_id": "child", "content_blocks": [{"type": "text", "text": "PRIVATE_NESTED_WRITER"}]},
      {"type": "tool_call_complete", "role": "writer", "result": "PRIVATE_TOOL_RESULT"},
      {"type": "approval_requested", "role": "writer", "payload": "PRIVATE_APPROVAL"},
      {"type": "assistant_message", "role": "writer", "content_blocks": [{"type": "thinking", "thinking": "PRIVATE_ONLY_THINKING"}]},
      {"type": "fork_seed", "messages": [{"role": "user", "content": "PRIVATE_MODEL_CONTEXT"}], "source_log_path": "/PRIVATE_SOURCE_PATH"},
    ]:
      log.append_sync(event)
    response = _history(client, init)

    assert response.status_code == 200, response.text
    assert response.json() == {
      "schema_version": 1,
      "entries": [
        {"seq": user.seq, "role": "user", "content": "Show earnings", "timestamp": user.timestamp},
        {"seq": assistant.seq, "role": "assistant", "content": "Revenue grew.", "timestamp": assistant.timestamp},
        {"seq": next_assistant.seq, "role": "assistant", "content": "More detail.", "timestamp": next_assistant.timestamp},
      ],
      "next_before_seq": None,
    }
    assert "PRIVATE_" not in response.text


def test_history_pages_newest_first_and_returns_each_page_ascending(tmp_path: Path) -> None:
  with TestClient(_make_app(tmp_path / "logs", _ReplayProvider([]), None)) as client:
    init = _post_init(client).json()
    log = _bind_log(client, init, tmp_path / "logs" / "pages.jsonl")
    for index in range(5):
      log.append_sync({"type": "user_message", "role": "writer", "client_kind": "excel", "content": f"Message {index}", "analyst_content": f"Message {index}"})
      log.append_sync({"type": "turn_complete", "role": "writer"})
    first = _history(client, init, limit=2).json()
    assert [entry["content"] for entry in first["entries"]] == ["Message 3", "Message 4"]
    assert first["next_before_seq"] == first["entries"][0]["seq"]

    # A new append must not move an already-issued older-page boundary.
    log.append_sync({"type": "user_message", "role": "writer", "client_kind": "excel", "content": "New message", "analyst_content": "New message"})
    second = _history(client, init, limit=2, before_seq=first["next_before_seq"]).json()
    assert [entry["content"] for entry in second["entries"]] == ["Message 1", "Message 2"]
    assert second["next_before_seq"] == second["entries"][0]["seq"]
    third = _history(client, init, limit=2, before_seq=second["next_before_seq"]).json()
    assert [entry["content"] for entry in third["entries"]] == ["Message 0"]
    assert third["next_before_seq"] is None
    assert _history(client, init, before_seq=third["entries"][0]["seq"]).json()["entries"] == []


def test_empty_projected_page_still_returns_older_cursor(tmp_path: Path) -> None:
  with TestClient(_make_app(tmp_path / "logs", _ReplayProvider([]), None)) as client:
    init = _post_init(client).json()
    log = _bind_log(client, init, tmp_path / "logs" / "invisible.jsonl")
    visible = log.append_sync({"type": "user_message", "role": "writer", "client_kind": "excel", "content": "Analyst", "analyst_content": "Analyst"})
    hidden = log.append_sync({"type": "user_message", "role": "writer", "client_kind": "operator", "content": "Internal", "analyst_content": "Internal"})
    page = _history(client, init, limit=1).json()
    assert page == {"schema_version": 1, "entries": [], "next_before_seq": hidden.seq}
    older = _history(client, init, limit=1, before_seq=page["next_before_seq"]).json()
    assert [entry["seq"] for entry in older["entries"]] == [visible.seq]
    assert older["next_before_seq"] is None


def test_history_default_and_maximum_page_sizes(tmp_path: Path) -> None:
  with TestClient(_make_app(tmp_path / "logs", _ReplayProvider([]), None)) as client:
    init = _post_init(client).json()
    log = _bind_log(client, init, tmp_path / "logs" / "bounded.jsonl")
    for index in range(201):
      log.append_sync({"type": "user_message", "role": "writer", "client_kind": "excel", "content": f"Message {index}", "analyst_content": f"Message {index}"})
    default = _history(client, init).json()
    assert len(default["entries"]) == 50
    assert default["entries"][0]["content"] == "Message 151"
    maximum = _history(client, init, limit=200).json()
    assert len(maximum["entries"]) == 200
    assert maximum["entries"][0]["content"] == "Message 1"


@pytest.mark.parametrize("params", [
  {"limit": 0}, {"limit": 201}, {"limit": "invalid"},
  {"before_seq": 0}, {"before_seq": "invalid"},
  {"user_id": "bob"}, {"session_id": "other"},
  {"log_session_id": "other"}, {"path": "/not/a/session/log"},
])
def test_history_query_contract_allows_only_bounded_paging(tmp_path: Path, params: dict[str, Any]) -> None:
  with TestClient(_make_app(tmp_path / "logs", _ReplayProvider([]), None)) as client:
    init = _post_init(client).json()
    response = _history(client, init, **params)
    assert response.status_code == 422, response.text


def test_history_requires_bearer_and_cannot_read_another_sessions_log(tmp_path: Path) -> None:
  with TestClient(_make_app(tmp_path / "logs", _ReplayProvider([]), None)) as client:
    alice = _post_init(client, user_id="alice").json()
    bob = _post_init(client, user_id="bob").json()
    for init, owner in [(alice, "Alice"), (bob, "Bob")]:
      log = _bind_log(client, init, tmp_path / "logs" / f"{owner}.jsonl")
      log.append_sync({"type": "user_message", "role": "writer", "client_kind": "excel", "content": f"{owner}'s private message", "analyst_content": f"{owner}'s private message"})
    assert client.get("/api/chat/history").status_code == 401
    assert client.get("/api/chat/history", headers={"Authorization": "Bearer invalid"}).status_code == 401
    assert [entry["content"] for entry in _history(client, alice).json()["entries"]] == ["Alice's private message"]
    assert [entry["content"] for entry in _history(client, bob).json()["entries"]] == ["Bob's private message"]
    selected = _history(client, alice, session_id=bob["session_id"])
    assert selected.status_code == 422
    assert "Bob's private message" not in selected.text
