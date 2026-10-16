"""Undo holders follow durable conversations, not authentication-session expiry."""

import asyncio
import inspect
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

TESTS_DIR = Path(__file__).resolve().parent
if str(TESTS_DIR) not in sys.path:
  sys.path.insert(0, str(TESTS_DIR))

import test_session_resume as resume_helpers  # noqa: E402
from agent_gateway import AgentRunner, StreamEvent, ToolDispatcher, ToolResultContext  # noqa: E402
from agent_gateway.session import GatewaySession  # noqa: E402
from agent_gateway.undo_token_lifetime import (  # noqa: E402
  UndoTokenLifetimeRegistry,
  conversation_id,
  observe_runner_results,
  owner_id,
  restore_tokens,
)
from test_session_resume import (  # noqa: E402
  _make_app,
  _post_init,
  _ready_session,
  _ReplayProvider,
  _ResumeResolver,
  _turn,
)


@pytest.fixture(autouse=True)
def _isolated_gateway_storage(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
  monkeypatch.setenv("AGENT_GATEWAY_AUTONOMOUS_LOG_DIR", str(tmp_path / "autonomous"))


@pytest.fixture
def registry(tmp_path: Path) -> UndoTokenLifetimeRegistry:
  return UndoTokenLifetimeRegistry(tmp_path / "undo.sqlite3")


def _session(session_id: str = "auth-one", *, user: str = "alice", tenant: str = "desk") -> GatewaySession:
  return GatewaySession(
    session_id=session_id, api_key_hash="hash", created_at=1, expires_at=2,
    user_id=user, owner_user_id=user, tenant_id=tenant,
  )


def _record(
  registry: UndoTokenLifetimeRegistry,
  session: GatewaySession,
  token: str,
) -> None:
  registry.record_tool_result(
    owner=owner_id(session), conversation=conversation_id(session),
    tool_name="write_cells", tool_input={}, result={"restore_token": token},
  )


def _headers(init: dict[str, Any]) -> dict[str, str]:
  return {"Authorization": f"Bearer {init['session_token']}"}


def _http_session(app: FastAPI, init: dict[str, Any]) -> GatewaySession:
  session = app.state.auth.session_store.get_session(init["session_id"])
  assert session is not None
  return session


def _statuses(
  client: TestClient,
  init: dict[str, Any],
  tokens: list[str],
  *,
  conversation: str | None = None,
) -> dict[str, Any]:
  payload: dict[str, Any] = {"tokens": tokens}
  if conversation is not None:
    payload["conversation_id"] = conversation
  response = client.post("/api/workbook-undo/status", headers=_headers(init), json=payload)
  assert response.status_code == 200, response.text
  return response.json()


@pytest.mark.parametrize(
  "result, expected",
  [
    ({"restore_token": "restore"}, {"restore"}),
    ({"undo_token": "undo"}, {"undo"}),
    ({"_undo_meta": "metadata-token"}, {"metadata-token"}),
    ({"_undo_meta": {"restore_token": "nested-meta"}}, {"nested-meta"}),
    ({"results": [{"value": ({"undo_token": "nested"},)}]}, {"nested"}),
    (
      {"content": [{"type": "text", "text": json.dumps({"data": {"restore_token": "mcp"}})}]},
      {"mcp"},
    ),
    ('  [{"undo_token": "json-list"}, {"restore_token": "json-list"}]', {"json-list"}),
    ({"content": [{"text": "{invalid JSON"}], "undo_token": "", "restore_token": None}, set()),
  ],
)
def test_nested_and_mcp_results_register_all_token_forms(
  registry: UndoTokenLifetimeRegistry, result: Any, expected: set[str],
) -> None:
  assert restore_tokens(result) == expected
  registry.record_tool_result(
    owner="owner", conversation="conversation", tool_name="write_cells",
    tool_input={}, result=result,
  )
  assert registry.statuses([*expected, "unobserved"], owner="owner") == {
    **{token: "active" for token in expected}, "unobserved": "unknown",
  }


def test_registry_reload_preserves_active_terminal_and_consumed_holders(
  registry: UndoTokenLifetimeRegistry,
) -> None:
  for conversation, token in [("live", "active"), ("closed", "terminal"), ("restored", "consumed")]:
    registry.record_tool_result(
      owner="owner", conversation=conversation, tool_name="write_cells",
      tool_input={}, result={"restore_token": token},
    )
  registry.close_conversation(owner="owner", conversation="closed")
  registry.record_tool_result(
    owner="owner", conversation="restored", tool_name="excel__restore_range",
    tool_input={"restore_token": "consumed"}, result={"restored": True},
  )

  reloaded = UndoTokenLifetimeRegistry(registry.path)
  assert reloaded.statuses(["active", "terminal", "consumed", "missing"], owner="owner") == {
    "active": "active", "terminal": "terminal", "consumed": "consumed", "missing": "unknown",
  }
  assert reloaded.is_closed(owner="owner", conversation="closed")
  assert not reloaded.is_closed(owner="other-owner", conversation="closed")
  # Late results cannot reopen an explicitly ended conversation.
  reloaded.record_tool_result(
    owner="owner", conversation="closed", tool_name="write_cells",
    tool_input={}, result={"undo_token": "late-result"},
  )
  assert reloaded.statuses(["late-result"], owner="owner") == {"late-result": "terminal"}


def test_begin_close_blocks_attachments_but_preserves_tokens_until_settled(
  registry: UndoTokenLifetimeRegistry,
) -> None:
  session = _session()
  owner = owner_id(session)
  conversation = conversation_id(session)
  _record(registry, session, "before-cancellation")
  registry.begin_close(owner=owner, conversation=conversation)
  assert registry.is_closed(owner=owner, conversation=conversation)
  assert registry.statuses(["before-cancellation"], owner=owner) == {
    "before-cancellation": "active",
  }
  # A running tool can still publish another backup while cancellation settles.
  _record(registry, session, "during-cancellation")
  reloaded = UndoTokenLifetimeRegistry(registry.path)
  assert reloaded.is_closed(owner=owner, conversation=conversation)
  assert reloaded.statuses(["before-cancellation", "during-cancellation"], owner=owner) == {
    "before-cancellation": "active", "during-cancellation": "active",
  }
  reloaded.close_conversation(owner=owner, conversation=conversation)
  # A duplicate begin-close must not reopen a fully settled conversation.
  reloaded.begin_close(owner=owner, conversation=conversation)
  assert reloaded.statuses(["before-cancellation", "during-cancellation"], owner=owner) == {
    "before-cancellation": "terminal", "during-cancellation": "terminal",
  }


@pytest.mark.parametrize("source_state", ["closing", "closed"])
def test_new_fork_cannot_inherit_tokens_from_an_ended_source(
  registry: UndoTokenLifetimeRegistry, tmp_path: Path, source_state: str,
) -> None:
  source = _session("source-auth")
  source.resume_log_path = tmp_path / "source.jsonl"
  _record(registry, source, "source-token")
  owner = owner_id(source)
  if source_state == "closing":
    registry.begin_close(owner=owner, conversation="source")
  else:
    registry.close_conversation(owner=owner, conversation="source")
  assert registry.fork_conversation(owner=owner, source="source", target="new-fork") is False
  assert registry.statuses(
    ["source-token"], owner=owner, conversation="new-fork",
  ) == {"source-token": "unknown"}

  fork = _session("new-fork-auth")
  fork.resume_log_path = tmp_path / "new-fork.jsonl"
  fork.fork_source_log_path = source.resume_log_path
  built: list[bool] = []

  def build_runner():
    built.append(True)
    return SimpleNamespace()

  runtime = SimpleNamespace(build_runner=build_runner)
  with pytest.raises(HTTPException) as refusal:
    observe_runner_results(runtime, fork, registry)
  assert refusal.value.status_code == 409
  assert refusal.value.detail == "Source conversation is closed"
  assert runtime.build_runner is build_runner
  assert built == []
  assert registry.statuses(["source-token"], owner=owner) == {
    "source-token": "active" if source_state == "closing" else "terminal",
  }


def test_established_fork_continues_after_source_close_without_recloning(
  registry: UndoTokenLifetimeRegistry, tmp_path: Path,
) -> None:
  source = _session("source-auth")
  source.resume_log_path = tmp_path / "source.jsonl"
  _record(registry, source, "inherited-token")
  fork = _session("fork-auth")
  fork.resume_log_path = tmp_path / "established-fork.jsonl"
  fork.fork_source_log_path = source.resume_log_path
  runtime = SimpleNamespace(build_runner=lambda: SimpleNamespace())
  observe_runner_results(runtime, fork, registry)
  owner = owner_id(source)
  assert registry.statuses(
    ["inherited-token"], owner=owner, conversation="established-fork",
  ) == {"inherited-token": "active"}

  _record(registry, source, "source-only-after-fork")
  registry.close_conversation(owner=owner, conversation="source")
  reloaded = UndoTokenLifetimeRegistry(registry.path)
  # Each later turn rebuilds the runtime, but must retain this fork's own history.
  continued_runtime = SimpleNamespace(build_runner=lambda: SimpleNamespace())
  observe_runner_results(continued_runtime, fork, reloaded)
  assert reloaded.fork_conversation(
    owner=owner, source="source", target="established-fork",
  ) is True
  assert reloaded.statuses(
    ["inherited-token", "source-only-after-fork"], owner=owner,
    conversation="established-fork",
  ) == {"inherited-token": "active", "source-only-after-fork": "unknown"}
  runner = continued_runtime.build_runner()
  ctx = SimpleNamespace(
    tool_name="write_cells", tool_input={}, result={"undo_token": "fork-next-token"}, error=None,
  )
  assert asyncio.run(runner._on_tool_result(ctx)) is None
  assert reloaded.statuses(
    ["fork-next-token"], owner=owner, conversation="established-fork",
  ) == {"fork-next-token": "active"}
  reloaded.close_conversation(owner=owner, conversation="established-fork")
  assert reloaded.statuses(["inherited-token"], owner=owner) == {"inherited-token": "terminal"}


@pytest.mark.parametrize("tool_name", ["restore_range", "excel__restore_range", "undo", "excel__undo_cells"])
def test_successful_restore_consumes_all_same_owner_holders_only(
  registry: UndoTokenLifetimeRegistry, tool_name: str,
) -> None:
  for owner, conversation in [("alice", "one"), ("alice", "two"), ("bob", "one")]:
    registry.record_tool_result(
      owner=owner, conversation=conversation, tool_name="write_cells",
      tool_input={}, result={"restore_token": "shared-token"},
    )
  registry.record_tool_result(
    owner="alice", conversation="two", tool_name=tool_name,
    tool_input={"restore_token": "shared-token"}, result={"restored": True},
  )
  assert registry.statuses(["shared-token"], owner="alice") == {"shared-token": "consumed"}
  assert registry.statuses(["shared-token"], owner="bob") == {"shared-token": "active"}
  registry.record_tool_result(
    owner="alice", conversation="later-holder", tool_name="write_cells",
    tool_input={}, result={"undo_token": "shared-token"},
  )
  reloaded = UndoTokenLifetimeRegistry(registry.path)
  assert reloaded.statuses(
    ["shared-token"], owner="alice", conversation="later-holder",
  ) == {"shared-token": "consumed"}


@pytest.mark.parametrize(
  "result, error",
  [
    ({"restored": False}, None),
    ({"success": False}, None),
    ({"isError": True, "content": [{"type": "text", "text": "restore failed"}]}, None),
    ({"error": {"code": "restore_failed"}}, None),
    (None, {"code": "transport_error"}),
  ],
)
def test_failed_restore_preserves_holder(
  registry: UndoTokenLifetimeRegistry, result: Any, error: Any,
) -> None:
  session = _session()
  _record(registry, session, "retryable")
  registry.record_tool_result(
    owner=owner_id(session), conversation=conversation_id(session),
    tool_name="excel__restore_range", tool_input={"restore_token": "retryable"},
    result=result, error=error,
  )
  assert registry.statuses(["retryable"], owner=owner_id(session)) == {"retryable": "active"}


def test_non_restore_input_does_not_consume_a_token(registry: UndoTokenLifetimeRegistry) -> None:
  session = _session()
  _record(registry, session, "still-live")
  registry.record_tool_result(
    owner=owner_id(session), conversation=conversation_id(session), tool_name="inspect_range",
    tool_input={"undo_token": "still-live"}, result={"success": True},
  )
  assert registry.statuses(["still-live"], owner=owner_id(session)) == {"still-live": "active"}


def test_durable_conversation_identity_uses_log_stem_not_auth_session(
  registry: UndoTokenLifetimeRegistry, tmp_path: Path,
) -> None:
  original = _session()
  original.resume_log_path = tmp_path / "first-location" / "durable-conversation.jsonl"
  refreshed = _session("new-auth")
  refreshed.resume_log_path = tmp_path / "other-location" / "durable-conversation.jsonl"
  _record(registry, original, "saved")
  assert conversation_id(original) == conversation_id(refreshed) == "durable-conversation"
  assert registry.statuses(
    ["saved"], owner=owner_id(refreshed), conversation=conversation_id(refreshed),
  ) == {"saved": "active"}
  assert owner_id(_session(tenant="other-desk")) != owner_id(original)
  assert registry.statuses(["saved"], owner=owner_id(_session(tenant="other-desk"))) == {"saved": "unknown"}


@pytest.mark.parametrize("async_previous", [False, True])
def test_observe_runner_results_preserves_hook_return_and_factory_signature(
  registry: UndoTokenLifetimeRegistry, tmp_path: Path, async_previous: bool,
) -> None:
  session = _session()
  seen: list[ToolResultContext] = []
  extras = [{"type": "text", "text": "existing hook result"}]

  def previous(ctx: ToolResultContext):
    seen.append(ctx)
    assert registry.statuses(["hook-token"], owner=owner_id(session)) == {"hook-token": "active"}
    return extras

  async def async_hook(ctx: ToolResultContext):
    return previous(ctx)

  built: list[Any] = []

  def build_runner(event_log: Any, session_id: str, started_at: float | None = None):
    runner = SimpleNamespace(_on_tool_result=async_hook if async_previous else previous)
    built.append((event_log, session_id, started_at, runner))
    return runner

  runtime = SimpleNamespace(build_runner=build_runner)
  original_signature = inspect.signature(runtime.build_runner)
  observe_runner_results(runtime, session, registry)
  assert inspect.signature(runtime.build_runner) == original_signature
  runner = runtime.build_runner("events", session.session_id, started_at=123.0)
  assert built == [("events", session.session_id, 123.0, runner)]
  # The runtime may bind its durable log after construction, before a result arrives.
  session.resume_log_path = tmp_path / "bound-after-construction.jsonl"
  ctx = ToolResultContext(
    tool_name="write_cells", tool_input={}, redacted_tool_input={},
    result={"content": [{"type": "text", "text": '{"undo_token": "hook-token"}'}]},
    error=None, duration_ms=1, tool_call_id="call-one", session_id=session.session_id,
    server=None, result_entry=None,
  )
  assert asyncio.run(runner._on_tool_result(ctx)) is extras
  assert seen == [ctx]
  assert registry.statuses(
    ["hook-token"], owner=owner_id(session), conversation="bound-after-construction",
  ) == {"hook-token": "active"}


def test_observe_runner_results_without_previous_hook_registers_result(
  registry: UndoTokenLifetimeRegistry,
) -> None:
  session = _session()
  runtime = SimpleNamespace(build_runner=lambda: SimpleNamespace())
  observe_runner_results(runtime, session, registry)
  runner = runtime.build_runner()
  ctx = SimpleNamespace(
    tool_name="write_cells", tool_input={}, result={"_undo_meta": "no-previous"}, error=None,
  )
  assert asyncio.run(runner._on_tool_result(ctx)) is None
  assert registry.statuses(["no-previous"], owner=owner_id(session)) == {"no-previous": "active"}


def test_http_status_unknown_and_principal_isolation(tmp_path: Path) -> None:
  provider = _ReplayProvider([])
  app = _make_app(tmp_path / "logs", provider, None)
  with TestClient(app) as client:
    alice = _post_init(client).json()
    bob = _post_init(client, user_id="bob").json()
    alice_session = _http_session(app, alice)
    alice_session.resume_log_path = tmp_path / "alice-log.jsonl"
    _record(app.state.undo_token_lifetimes, alice_session, "alice-only")
    assert _statuses(client, alice, ["alice-only", "missing"]) == {
      "conversation_id": "alice-log", "statuses": {"alice-only": "active", "missing": "unknown"},
    }
    assert _statuses(client, bob, ["alice-only"], conversation="alice-log")["statuses"] == {
      "alice-only": "unknown",
    }
    # Even guessing another owner's conversation id cannot retire their holders.
    response = client.delete("/api/chat/conversations/alice-log", headers=_headers(bob))
    assert response.status_code == 200, response.text
    assert _statuses(client, alice, ["alice-only"])["statuses"] == {"alice-only": "active"}
    assert client.post("/api/workbook-undo/status", json={"tokens": ["alice-only"]}).status_code == 401
  assert provider.captured_messages == []


def test_http_conversation_filter_cannot_hide_another_live_holder(tmp_path: Path) -> None:
  app = _make_app(tmp_path / "logs", _ReplayProvider([]), None)
  with TestClient(app) as client:
    closed = _post_init(client).json()
    live = _post_init(client).json()
    for init, name in [(closed, "closed-holder"), (live, "live-holder")]:
      session = _http_session(app, init)
      session.resume_log_path = tmp_path / f"{name}.jsonl"
      _record(app.state.undo_token_lifetimes, session, "shared")
    response = client.delete("/api/chat/conversations/closed-holder", headers=_headers(live))
    assert response.status_code == 200, response.text
    assert _statuses(client, live, ["shared"], conversation="closed-holder")["statuses"] == {"shared": "active"}
    assert _statuses(client, live, ["shared"], conversation="unrelated")["statuses"] == {"shared": "unknown"}
    assert _statuses(client, live, ["shared"])["statuses"] == {"shared": "active"}


def test_http_auth_refresh_and_resume_keep_durable_token_active(tmp_path: Path) -> None:
  resolver = _ResumeResolver()
  provider = _ReplayProvider([])
  app = _make_app(tmp_path / "logs", provider, resolver)
  with TestClient(app) as client:
    original = _post_init(client).json()
    session = _http_session(app, original)
    path = tmp_path / "logs" / "durable-resume.jsonl"
    session.resume_log_path = path
    _record(app.state.undo_token_lifetimes, session, "survives-auth")
    refreshed = {**original, "session_token": app.state.auth.issue_token(session)}
    assert _statuses(client, refreshed, ["survives-auth"])["statuses"] == {"survives-auth": "active"}
    app.state.auth.session_store.expire_session(session.session_id)
    resolver.outcomes[session.session_id] = path
    response = _post_init(client, resume_session_id=session.session_id)
    assert response.status_code == 200, response.text
    resumed = response.json()
    assert resumed["session_id"] != original["session_id"]
    assert _http_session(app, resumed).resume_log_path == path
    assert _statuses(client, resumed, ["survives-auth"]) == {
      "conversation_id": path.stem, "statuses": {"survives-auth": "active"},
    }
    assert client.post(
      "/api/workbook-undo/status", headers=_headers(original), json={"tokens": ["survives-auth"]},
    ).status_code == 401
  assert provider.captured_messages == []


@pytest.mark.parametrize("end_method", ["close", "delete"])
def test_http_explicit_end_expires_auth_and_refuses_resume(
  tmp_path: Path, end_method: str,
) -> None:
  resolver = _ResumeResolver()
  provider = _ReplayProvider([])
  app = _make_app(tmp_path / "logs", provider, resolver)
  with TestClient(app) as client:
    original = _post_init(client).json()
    session = _http_session(app, original)
    path = tmp_path / "logs" / "explicitly-ended.jsonl"
    session.resume_log_path = path
    _record(app.state.undo_token_lifetimes, session, "ended-token")
    resolver.outcomes[session.session_id] = path
    if end_method == "close":
      response = client.post("/api/chat/close", headers=_headers(original))
    else:
      response = client.delete(f"/api/chat/conversations/{path.stem}", headers=_headers(original))
    assert response.status_code == 200, response.text
    assert response.json() == {"conversation_id": path.stem, "status": "terminal"}
    assert app.state.auth.session_store.get_session(session.session_id) is None
    assert client.post(
      "/api/workbook-undo/status", headers=_headers(original), json={"tokens": ["ended-token"]},
    ).status_code == 401
    fresh = _post_init(client).json()
    assert fresh["session_id"] != original["session_id"]
    assert _statuses(client, fresh, ["ended-token", "unobserved"])["statuses"] == {
      "ended-token": "terminal", "unobserved": "unknown",
    }
    refused = _post_init(client, resume_session_id=session.session_id)
    assert refused.status_code == 409, refused.text
    assert refused.json() == {"error": "session_resume_refused", "reason": "not_found"}
  assert provider.captured_messages == []


@pytest.mark.parametrize("scoped_wrapper", [False, True])
def test_real_gateway_runner_registers_tool_result_before_status_query(
  tmp_path: Path, monkeypatch: pytest.MonkeyPatch, scoped_wrapper: bool,
) -> None:
  tool_name = "lookup_workbook_snapshot"
  calls: list[dict[str, Any]] = []

  async def handler(tool_input: dict[str, Any], **_kwargs: Any):
    calls.append(tool_input)
    return {"content": [{"type": "text", "text": '{"restore_token": "runner-issued"}'}]}, None

  def dispatcher_factory(**kwargs: Any):
    kwargs["local_tool_handlers"] = {tool_name: handler}
    kwargs["role"] = "owner"
    return ToolDispatcher(**kwargs)

  def runner_factory(**kwargs: Any):
    kwargs["get_tool_definitions"] = lambda: [{
      "name": tool_name, "description": "Read a workbook snapshot",
      "input_schema": {"type": "object", "properties": {}},
    }]
    runner = AgentRunner(**kwargs)
    return _ForwardingRunner(runner) if scoped_wrapper else runner

  class ToolProvider(_ReplayProvider):
    def __init__(self) -> None:
      super().__init__(["Snapshot ready."])
      self.sent_tool = False

    async def stream(self, client: Any, params: dict[str, Any]):
      if not self.sent_tool:
        self.sent_tool = True
        yield StreamEvent(type="message_start", input_tokens=10)
        yield StreamEvent(type="tool_use_start", tool_id="snapshot-call", tool_name=tool_name)
        yield StreamEvent(
          type="tool_use_end", tool_id="snapshot-call", tool_name=tool_name,
          tool_input={}, raw_block={"type": "tool_use", "id": "snapshot-call", "name": tool_name, "input": {}},
        )
        yield StreamEvent(type="message_end", stop_reason="tool_use")
      else:
        async for event in super().stream(client, params):
          yield event

  monkeypatch.setattr(resume_helpers, "ToolDispatcher", dispatcher_factory)
  monkeypatch.setattr(resume_helpers, "AgentRunner", runner_factory)
  provider = ToolProvider()
  app = _make_app(tmp_path / "logs", provider, None)
  with TestClient(app) as client:
    init = _ready_session(client, _post_init(client).json())
    assert _statuses(client, init, ["runner-issued"])["statuses"] == {"runner-issued": "unknown"}
    _turn(client, init, "Read the workbook snapshot.")
    assert calls == [{}]
    session = _http_session(app, init)
    assert session.agent_session_log is not None
    assert _statuses(client, init, ["runner-issued"]) == {
      "conversation_id": session.agent_session_log.path.stem,
      "statuses": {"runner-issued": "active"},
    }
    assert len(provider.captured_messages) == 2


class _ForwardingRunner:
  """The product's UserScopedRunner shape: reads forward, writes land on the wrapper."""

  def __init__(self, runner: Any) -> None:
    self._runner = runner

  async def run(self, *args: Any, **kwargs: Any) -> Any:
    return await self._runner.run(*args, **kwargs)

  def __getattr__(self, name: str) -> Any:
    return getattr(self._runner, name)


class _InnerForwardingRunner:
  """The product's RunnerWithCitationValidation shape: inner runner lives on _inner."""

  def __init__(self, runner: Any) -> None:
    self._inner = runner

  async def run(self, *args: Any, **kwargs: Any) -> Any:
    return await self._inner.run(*args, **kwargs)

  def __getattr__(self, name: str) -> Any:
    return getattr(self._inner, name)


@pytest.mark.parametrize("wrappers", [1, 2])
def test_observe_runner_results_reaches_the_runner_behind_forwarding_wrappers(
  registry: UndoTokenLifetimeRegistry, wrappers: int,
) -> None:
  session = _session()
  order: list[str] = []
  extras = [{"type": "text", "text": "product block"}]

  async def product_hook(ctx: Any):
    order.append("product")
    return extras

  inner = SimpleNamespace(_on_tool_result=product_hook)

  def build_runner() -> Any:
    built: Any = inner
    if wrappers >= 2:
      built = _InnerForwardingRunner(built)
    built = _ForwardingRunner(built)
    return built

  runtime = SimpleNamespace(build_runner=build_runner)
  observe_runner_results(runtime, session, registry)
  runner = runtime.build_runner()
  # The real runner reads its own attribute, never the wrapper's.
  assert inner._on_tool_result is not product_hook
  ctx = SimpleNamespace(
    tool_name="write_cells", tool_input={}, result={"restore_token": "wrapped"}, error=None,
  )
  assert asyncio.run(inner._on_tool_result(ctx)) is extras
  assert order == ["product"]
  assert registry.statuses(["wrapped"], owner=owner_id(session)) == {"wrapped": "active"}
  assert runner._on_tool_result is inner._on_tool_result


def test_registry_failure_does_not_skip_the_product_hook(
  registry: UndoTokenLifetimeRegistry, monkeypatch: pytest.MonkeyPatch,
) -> None:
  session = _session()
  seen: list[Any] = []

  def product_hook(ctx: Any):
    seen.append(ctx)
    return "product-result"

  def broken(**_kwargs: Any) -> None:
    raise RuntimeError("sqlite is unavailable")

  monkeypatch.setattr(registry, "record_tool_result", broken)
  runtime = SimpleNamespace(build_runner=lambda: SimpleNamespace(_on_tool_result=product_hook))
  observe_runner_results(runtime, session, registry)
  runner = runtime.build_runner()
  ctx = SimpleNamespace(tool_name="write_cells", tool_input={}, result={"restore_token": "t"}, error=None)
  assert asyncio.run(runner._on_tool_result(ctx)) == "product-result"
  assert seen == [ctx]


class _FakeApprovalStore:
  def __init__(self, requests: list[Any]) -> None:
    self.requests = {request.approval_id: request for request in requests}
    self.cancelled: list[str] = []
    self.identities: dict[str, dict[str, Any]] = {}

  async def list_parked_requests(self, tenant_storage_id: str) -> list[Any]:
    return [r for r in self.requests.values() if r.approval_id not in self.cancelled]

  async def get(self, approval_id: str) -> Any:
    return self.requests.get(approval_id)

  async def fence_persistent_grants_for_cancellation(self, approval_id: str, **identity: Any):
    return self.requests[approval_id], True

  async def terminalize_pending_for_cancellation(self, approval_id: str, **identity: Any):
    self.cancelled.append(approval_id)
    self.identities[approval_id] = identity
    return self.requests[approval_id], True, True

  async def revoke_persistent_grants_for_approval(self, approval_id: str) -> int:
    return 0


def _parked(approval_id: str, session_id: str) -> Any:
  return SimpleNamespace(
    approval_id=approval_id, tool_call_id=f"call-{approval_id}", user_id="alice",
    request_id=f"req-{approval_id}", run_id=f"run-{approval_id}", session_id=session_id,
    channel="excel",
  )


def test_cancel_parked_approvals_denies_only_the_closed_conversations_sessions() -> None:
  from agent_gateway.undo_token_lifetime import cancel_parked_approvals

  store = _FakeApprovalStore([_parked("closed", "job-closed"), _parked("other", "job-other")])
  resolved: list[str] = []

  async def on_resolve(*, request: Any) -> None:
    resolved.append(request.approval_id)

  policy = SimpleNamespace(on_resolve=on_resolve)
  cancelled = asyncio.run(cancel_parked_approvals(
    store, policy, tenant_storage_id="alice", session_ids={"job-closed"},
  ))
  assert cancelled == 1
  assert store.cancelled == ["closed"]
  assert resolved == ["closed"]
  assert store.identities["closed"]["expected_session_id"] == "job-closed"
  assert store.identities["closed"]["decision_reason"] == "conversation_closed"


def test_session_attachment_survives_for_closed_conversation_lookups(
  registry: UndoTokenLifetimeRegistry,
) -> None:
  session = _session()
  runtime = SimpleNamespace(build_runner=lambda: SimpleNamespace())
  observe_runner_results(runtime, session, registry)
  runtime.build_runner()
  owner, conversation = owner_id(session), conversation_id(session)
  assert registry.session_ids(owner=owner, conversation=conversation) == {session.session_id}
  assert not registry.is_session_closed(owner=owner, session_id=session.session_id)
  registry.begin_close(owner=owner, conversation=conversation)
  assert registry.is_session_closed(owner=owner, session_id=session.session_id)
  assert not registry.is_session_closed(owner=owner_id(_session(user="bob")), session_id=session.session_id)


def test_guarded_root_continue_refuses_closed_conversations(
  registry: UndoTokenLifetimeRegistry,
) -> None:
  from agent_gateway.approval_continuation import RootContinueOutcome, RootContinueRequest
  from agent_gateway.undo_token_lifetime import guard_root_continue

  session = _session()
  runtime = SimpleNamespace(build_runner=lambda: SimpleNamespace())
  observe_runner_results(runtime, session, registry)
  runtime.build_runner()
  store = _FakeApprovalStore([_parked("parked", session.session_id), _parked("elsewhere", "job-unknown")])
  app = SimpleNamespace(state=SimpleNamespace(gateway_approval_store=store, undo_token_lifetimes=registry))
  delivered: list[str] = []

  async def hook(continue_request: Any) -> RootContinueOutcome:
    delivered.append(continue_request.approval_id)
    return RootContinueOutcome("delivered")

  guarded = guard_root_continue(hook, app, "desk")
  assert guard_root_continue(guarded, app, "desk") is guarded
  assert guard_root_continue(None, app, "desk") is None
  request = RootContinueRequest(approval_id="parked", tenant_storage_id="alice")
  assert asyncio.run(guarded(request)).status == "delivered"
  registry.begin_close(owner=owner_id(session), conversation=conversation_id(session))
  assert asyncio.run(guarded(request)).status == "not_restorable"
  elsewhere = RootContinueRequest(approval_id="elsewhere", tenant_storage_id="alice")
  assert asyncio.run(guarded(elsewhere)).status == "delivered"
  assert delivered == ["parked", "elsewhere"]


def test_http_close_denies_parked_approvals_of_the_closed_conversation(tmp_path: Path) -> None:
  provider = _ReplayProvider([])
  app = _make_app(tmp_path / "logs", provider, None)
  with TestClient(app) as client:
    init = _post_init(client).json()
    session = _http_session(app, init)
    registry = app.state.undo_token_lifetimes
    # An earlier gateway session of the same conversation already expired.
    registry.attach_session(
      owner=owner_id(session), session_id="earlier-job", conversation=conversation_id(session),
    )
    store = _FakeApprovalStore([
      _parked("live", session.session_id), _parked("earlier", "earlier-job"),
      _parked("unrelated", "someone-else"),
    ])
    resolved: list[str] = []

    async def on_resolve(*, request: Any) -> None:
      resolved.append(request.approval_id)

    app.state.gateway_approval_store = store
    app.state.gateway_approval_policy = SimpleNamespace(on_resolve=on_resolve)
    response = client.post("/api/chat/close", headers=_headers(init))
    assert response.status_code == 200, response.text
    assert sorted(store.cancelled) == ["earlier", "live"]
    assert sorted(resolved) == ["earlier", "live"]
    assert registry.is_closed(owner=owner_id(session), conversation=conversation_id(session))
  assert provider.captured_messages == []
