from __future__ import annotations

import asyncio
import json
import socket
import sys
import threading
import time
import types
from typing import Any

import httpx
import pytest

import agent_gateway.server as gateway_server
from agent_gateway.event_log import EventLog
from agent_gateway.providers import AnthropicProvider, OpenAIProvider
from agent_gateway.providers.agent_sdk import AgentSDKConfig, SDK_PINNED_VERSION
from agent_gateway.providers.base import ModelInfo, ModelProvider, StreamEvent
from agent_gateway.runner import AgentRunner
from agent_gateway.sdk_runner import AgentSDKRunner
from agent_gateway.server import ChatRuntime
from agent_gateway.tool_dispatcher import ToolDispatcher
from starlette.requests import ClientDisconnect


def _run(coro):
  return asyncio.run(coro)


def _chat_payload() -> dict[str, Any]:
  return {"messages": [{"role": "user", "content": "hello"}], "context": {}}


@pytest.fixture(autouse=True)
def _patch_asgi_transport_streaming(monkeypatch: pytest.MonkeyPatch):
  sentinel = object()

  class _StreamingResponseStream(httpx.AsyncByteStream):
    def __init__(
      self,
      body_queue: asyncio.Queue[Any],
      *,
      response_closed: asyncio.Event,
      app_task: asyncio.Task[Any],
    ) -> None:
      self._body_queue = body_queue
      self._response_closed = response_closed
      self._app_task = app_task
      self._closed = False

    async def __aiter__(self):
      try:
        while True:
          chunk = await self._body_queue.get()
          if chunk is sentinel:
            break
          yield chunk
      finally:
        await self.aclose()

    async def aclose(self) -> None:
      if self._closed:
        return
      self._closed = True
      self._response_closed.set()
      try:
        await self._app_task
      except (ClientDisconnect, OSError):
        pass

  async def _handle_async_request(self, request: httpx.Request) -> httpx.Response:
    assert isinstance(request.stream, httpx.AsyncByteStream)

    scope = {
      "type": "http",
      "asgi": {"version": "3.0", "spec_version": "2.3"},
      "http_version": "1.1",
      "method": request.method,
      "headers": [(k.lower(), v) for (k, v) in request.headers.raw],
      "scheme": request.url.scheme,
      "path": request.url.path,
      "raw_path": request.url.raw_path.split(b"?")[0],
      "query_string": request.url.query,
      "server": (request.url.host, request.url.port),
      "client": self.client,
      "root_path": self.root_path,
    }

    request_body = request.stream.__aiter__()
    request_complete = False
    response_started = asyncio.Event()
    response_closed = asyncio.Event()
    body_queue: asyncio.Queue[Any] = asyncio.Queue()
    status_code: int | None = None
    response_headers: list[tuple[bytes, bytes]] | None = None

    async def receive() -> dict[str, Any]:
      nonlocal request_complete

      if request_complete:
        await response_closed.wait()
        return {"type": "http.disconnect"}

      try:
        body = await request_body.__anext__()
      except StopAsyncIteration:
        request_complete = True
        return {"type": "http.request", "body": b"", "more_body": False}
      return {"type": "http.request", "body": body, "more_body": True}

    async def send(message: dict[str, Any]) -> None:
      nonlocal status_code, response_headers

      if message["type"] == "http.response.start":
        status_code = message["status"]
        response_headers = message.get("headers", [])
        response_started.set()
        return

      if message["type"] == "http.response.body":
        if response_closed.is_set():
          raise OSError("response stream closed")
        body = message.get("body", b"")
        more_body = message.get("more_body", False)
        if body and request.method != "HEAD":
          await body_queue.put(body)
        if not more_body:
          await body_queue.put(sentinel)

    async def _run_app() -> None:
      try:
        await self.app(scope, receive, send)
      finally:
        response_started.set()
        await body_queue.put(sentinel)

    app_task = asyncio.create_task(_run_app())
    await response_started.wait()
    if app_task.done():
      await app_task

    assert status_code is not None
    assert response_headers is not None

    return httpx.Response(
      status_code,
      headers=response_headers,
      stream=_StreamingResponseStream(
        body_queue,
        response_closed=response_closed,
        app_task=app_task,
      ),
    )

  monkeypatch.setattr(httpx.ASGITransport, "handle_async_request", _handle_async_request)


async def _init_session(client: httpx.AsyncClient) -> dict[str, Any]:
  response = await client.post("/api/chat/init", json={"api_key": "gateway-key"})
  assert response.status_code == 200, response.text
  return response.json()


async def _read_sse_event(response: httpx.Response) -> dict[str, Any]:
  async for line in response.aiter_lines():
    if line.startswith("data: "):
      return json.loads(line[6:])
  raise AssertionError("Expected at least one SSE event")


async def _collect_sse_events(response: httpx.Response) -> list[dict[str, Any]]:
  events: list[dict[str, Any]] = []
  async for line in response.aiter_lines():
    if line.startswith("data: "):
      events.append(json.loads(line[6:]))
  return events


async def _read_until_event_type(response: httpx.Response, event_type: str) -> dict[str, Any]:
  async for line in response.aiter_lines():
    if not line.startswith("data: "):
      continue
    event = json.loads(line[6:])
    if event.get("type") == event_type:
      return event
  raise AssertionError(f"Expected SSE event type {event_type!r}")


async def _wait_for(predicate, timeout: float, interval: float = 0.05) -> float:
  started = time.perf_counter()
  while True:
    if predicate():
      return time.perf_counter() - started
    elapsed = time.perf_counter() - started
    if elapsed >= timeout:
      raise AssertionError(f"Condition not met within {timeout:.2f}s")
    await asyncio.sleep(interval)


def _find_free_port() -> int:
  with socket.socket() as sock:
    sock.bind(("127.0.0.1", 0))
    return int(sock.getsockname()[1])


class _NullMcpClient:
  def is_mcp_tool(self, _name: str) -> bool:
    return False

  async def call_tool(self, name: str, _tool_input: dict[str, Any]):
    return None, {"code": "unknown_tool", "message": f"Unknown tool: {name}"}

  def get_tool_definitions(self) -> list[dict[str, Any]]:
    return []

  def get_server_for_tool(self, _name: str) -> str | None:
    return None


class _TrackingClient:
  def __init__(self) -> None:
    self.closed = False


class _DisconnectTrackingProvider(ModelProvider):
  name = "disconnect-tracking"

  def __init__(self) -> None:
    self.closed_clients: list[_TrackingClient] = []

  def has_active_credential(self, config: dict[str, Any]) -> bool:
    return True

  def create_client(self, config: dict[str, Any], *, timeout: float | None = None) -> Any:
    _ = config, timeout
    return _TrackingClient()

  async def close_client(self, client: Any, timeout: float = 2.0) -> None:
    _ = timeout
    client.closed = True
    self.closed_clients.append(client)

  def get_model_info(self, model: str) -> ModelInfo:
    return ModelInfo(id=model, provider=self.name)

  def build_request_params(
    self,
    *,
    model: str,
    messages: list[dict[str, Any]],
    system_prompt: str | list[tuple[str, bool]] | None,
    tools: list[dict[str, Any]],
    max_tokens: int,
    **kwargs: Any,
  ) -> dict[str, Any]:
    _ = model, messages, system_prompt, tools, max_tokens, kwargs
    return {}

  async def stream(self, client: Any, params: dict[str, Any]):
    _ = client, params
    if False:
      yield


class _ObservedAgentRunner(AgentRunner):
  def __init__(self, *args: Any, **kwargs: Any) -> None:
    super().__init__(*args, **kwargs)
    self.cancelled_calls = 0

  async def run(
    self,
    messages: list[dict[str, Any]],
    system_prompt: str | list[tuple[str, bool]] | None = None,
    model_override: str | None = None,
    max_turns: int | None = None,
  ) -> None:
    try:
      await super().run(
        messages=messages,
        system_prompt=system_prompt,
        model_override=model_override,
        max_turns=max_turns,
      )
    except asyncio.CancelledError:
      self.cancelled_calls += 1
      raise


class _RetryableDisconnectError(Exception):
  pass


class _HangingClient:
  def __init__(self) -> None:
    self.closed = False
    self.release = asyncio.Event()


class _HangingProvider(ModelProvider):
  name = "hanging"

  def __init__(self, *, text: str = "hello", retry_on_close: bool = True) -> None:
    self.text = text
    self.retry_on_close = retry_on_close
    self.close_calls = 0

  def has_active_credential(self, config: dict[str, Any]) -> bool:
    return True

  def create_client(self, config: dict[str, Any], *, timeout: float | None = None) -> Any:
    _ = config, timeout
    return _HangingClient()

  async def close_client(self, client: Any, timeout: float = 2.0) -> None:
    _ = timeout
    self.close_calls += 1
    client.closed = True
    client.release.set()

  def get_model_info(self, model: str) -> ModelInfo:
    return ModelInfo(id=model, provider=self.name)

  def build_request_params(
    self,
    *,
    model: str,
    messages: list[dict[str, Any]],
    system_prompt: str | list[tuple[str, bool]] | None,
    tools: list[dict[str, Any]],
    max_tokens: int,
    **kwargs: Any,
  ) -> dict[str, Any]:
    _ = model, messages, system_prompt, tools, max_tokens, kwargs
    return {}

  async def stream(self, client: Any, params: dict[str, Any]):
    _ = params
    yield StreamEvent(type="message_start", input_tokens=1)
    yield StreamEvent(type="text_delta", text=self.text)
    await client.release.wait()
    if self.retry_on_close:
      raise _RetryableDisconnectError("stream closed")

  def is_retryable_error(self, exc: Exception) -> bool:
    return isinstance(exc, _RetryableDisconnectError)


class _CompletingProvider(ModelProvider):
  name = "complete"

  def __init__(self, *, text: str = "done") -> None:
    self.text = text
    self.close_calls = 0

  def has_active_credential(self, config: dict[str, Any]) -> bool:
    return True

  def create_client(self, config: dict[str, Any], *, timeout: float | None = None) -> Any:
    _ = config, timeout
    return _TrackingClient()

  async def close_client(self, client: Any, timeout: float = 2.0) -> None:
    _ = timeout
    self.close_calls += 1
    client.closed = True

  def get_model_info(self, model: str) -> ModelInfo:
    return ModelInfo(id=model, provider=self.name)

  def build_request_params(
    self,
    *,
    model: str,
    messages: list[dict[str, Any]],
    system_prompt: str | list[tuple[str, bool]] | None,
    tools: list[dict[str, Any]],
    max_tokens: int,
    **kwargs: Any,
  ) -> dict[str, Any]:
    _ = model, messages, system_prompt, tools, max_tokens, kwargs
    return {}

  async def stream(self, client: Any, params: dict[str, Any]):
    _ = client, params
    yield StreamEvent(type="message_start", input_tokens=1)
    yield StreamEvent(type="text_delta", text=self.text)
    yield StreamEvent(type="text_end", raw_block={"type": "text", "text": self.text})
    yield StreamEvent(type="usage_update", output_tokens=1)
    yield StreamEvent(type="message_end", stop_reason="end_turn")


class _ToolCallProvider(ModelProvider):
  name = "tool-call"

  def has_active_credential(self, config: dict[str, Any]) -> bool:
    return True

  def create_client(self, config: dict[str, Any], *, timeout: float | None = None) -> Any:
    _ = config, timeout
    return _TrackingClient()

  async def close_client(self, client: Any, timeout: float = 2.0) -> None:
    _ = timeout
    client.closed = True

  def get_model_info(self, model: str) -> ModelInfo:
    return ModelInfo(id=model, provider=self.name)

  def build_request_params(
    self,
    *,
    model: str,
    messages: list[dict[str, Any]],
    system_prompt: str | list[tuple[str, bool]] | None,
    tools: list[dict[str, Any]],
    max_tokens: int,
    **kwargs: Any,
  ) -> dict[str, Any]:
    _ = model, messages, system_prompt, tools, max_tokens, kwargs
    return {}

  async def stream(self, client: Any, params: dict[str, Any]):
    _ = client, params
    yield StreamEvent(type="message_start", input_tokens=1)
    yield StreamEvent(type="tool_use_end", tool_id="tool_1", tool_name="hang_tool", tool_input={})
    yield StreamEvent(type="message_end", stop_reason="end_turn")


class _HangingDispatcher:
  def __init__(self, *, cancel_delay: float) -> None:
    self.cancel_delay = cancel_delay
    self.dispatch_calls = 0
    self.cancelled_calls = 0

  async def dispatch(
    self,
    tool_call_id: str,
    tool_name: str,
    tool_input: dict[str, Any],
    *,
    call_index: int = 0,
  ):
    _ = tool_call_id, tool_name, tool_input, call_index
    self.dispatch_calls += 1
    try:
      await asyncio.Future()
    except asyncio.CancelledError:
      self.cancelled_calls += 1
      await asyncio.sleep(self.cancel_delay)
      raise

  def requires_approval(self, tool_name: str, tool_input: dict[str, Any]) -> bool:
    _ = tool_name, tool_input
    return False


class _ManualAsyncIterator:
  def __init__(self) -> None:
    self.close_calls = 0
    self.closed = False

  async def aclose(self) -> None:
    self.close_calls += 1
    self.closed = True


class _HangingQueryIterator:
  def __init__(self) -> None:
    self.close_calls = 0
    self.closed = False
    self._released = asyncio.Event()
    self._sent_first = False

  def __aiter__(self):
    return self

  async def __anext__(self):
    if not self._sent_first:
      self._sent_first = True
      return types.SimpleNamespace(event={"type": "content_block_delta", "delta": {"type": "text_delta", "text": "sdk"}})
    await self._released.wait()
    raise StopAsyncIteration

  async def aclose(self) -> None:
    self.close_calls += 1
    self.closed = True
    self._released.set()


def _install_fake_agent_sdk(monkeypatch: pytest.MonkeyPatch, iterator_factory) -> None:
  class _HookMatcher:
    def __init__(self, *, hooks: list[Any]) -> None:
      self.hooks = hooks

  class _ClaudeAgentOptions:
    def __init__(self, **kwargs: Any) -> None:
      self.kwargs = kwargs

  module = types.ModuleType("claude_agent_sdk")
  module.__version__ = SDK_PINNED_VERSION
  module.HookMatcher = _HookMatcher
  module.ClaudeAgentOptions = _ClaudeAgentOptions
  module.query = lambda prompt, options: iterator_factory(prompt, options)
  monkeypatch.setitem(sys.modules, "claude_agent_sdk", module)


def _make_dispatcher(event_log: EventLog | None = None) -> ToolDispatcher:
  return ToolDispatcher(
    mcp_client=_NullMcpClient(),
    local_tool_handlers={},
    event_log=event_log or EventLog(),
    session_id="sess-stream-lifecycle",
  )


def test_close_client_handles_real_anthropic_contract() -> None:
  async def case() -> None:
    anthropic = pytest.importorskip("anthropic")
    provider = AnthropicProvider()
    client = anthropic.AsyncAnthropic(api_key="test-key")

    assert hasattr(client, "close")
    assert not hasattr(client, "aclose")
    assert getattr(client, "_client").is_closed is False

    await provider.close_client(client)

    assert getattr(client, "_client").is_closed is True

  _run(case())


def test_close_client_handles_real_openai_contract() -> None:
  async def case() -> None:
    openai = pytest.importorskip("openai")
    provider = OpenAIProvider()
    client = openai.AsyncOpenAI(api_key="test-key")

    assert hasattr(client, "close")
    assert not hasattr(client, "aclose")
    assert getattr(client, "_client").is_closed is False

    await provider.close_client(client)

    assert getattr(client, "_client").is_closed is True

  _run(case())


def test_disconnect_cleanup_fires_fast(make_test_app) -> None:
  async def case() -> None:
    provider = _HangingProvider()
    app = make_test_app(provider=provider, runner_class=_ObservedAgentRunner)

    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
      session_info = await _init_session(client)
      session = app.state.auth.session_store.get_session(session_info["session_id"])
      assert session is not None

      async with client.stream(
        "POST",
        "/api/chat",
        headers={"Authorization": f"Bearer {session_info['session_token']}"},
        json=_chat_payload(),
      ) as response:
        assert response.status_code == 200
        first_event = await _read_sse_event(response)
        assert first_event["type"] == "text_delta"

      elapsed = await _wait_for(
        lambda: app.state.test_state.disconnect_hook_calls == 1
        and getattr(app.state.test_state.runner, "cancelled_calls", 0) == 1
        and provider.close_calls >= 1
        and session.stream_active is False,
        timeout=1.0,
      )

      assert elapsed < 1.0

  _run(case())


def test_no_double_disconnect_firing(make_test_app) -> None:
  async def case() -> None:
    app = make_test_app(provider=_HangingProvider())

    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
      session_info = await _init_session(client)

      async with client.stream(
        "POST",
        "/api/chat",
        headers={"Authorization": f"Bearer {session_info['session_token']}"},
        json=_chat_payload(),
      ) as response:
        assert response.status_code == 200
        await _read_sse_event(response)

      await _wait_for(lambda: app.state.test_state.disconnect_hook_calls == 1, timeout=1.0)
      await app.state.test_state.runtime.on_disconnect()

      assert app.state.test_state.disconnect_hook_calls == 1

  _run(case())


def test_on_disconnect_default_no_op() -> None:
  runtime = ChatRuntime(system_prompt="test", build_runner=lambda _event_log, _sid: None)

  _run(runtime.on_disconnect())
  _run(runtime.on_disconnect())

  assert runtime._disconnect_called is True


def test_on_disconnect_agent_runner_closes_client() -> None:
  async def case() -> None:
    provider = _DisconnectTrackingProvider()
    runner = AgentRunner(
      event_log=EventLog(),
      dispatcher=_make_dispatcher(),
      session_id="sess-disconnect",
      provider=provider,
      auth_config={"api_key": "test-key", "model": "claude-sonnet-4-6"},
    )
    client = provider.create_client({}, timeout=None)
    runner._set_client(client)
    runtime = ChatRuntime(
      system_prompt="test",
      build_runner=lambda _event_log, _sid: runner,
      disconnect_handler=runner.on_disconnect,
    )

    await runtime.on_disconnect()

    assert runner._disconnected is True
    assert runner._active_client is None
    assert client.closed is True
    assert provider.closed_clients == [client]

  _run(case())


def test_on_disconnect_sdk_runner_closes_iterator() -> None:
  async def case() -> None:
    runner = AgentSDKRunner(
      event_log=EventLog(),
      session_id="sess-sdk",
      sdk_config=AgentSDKConfig(api_key="test-key", model="claude-sonnet-4-6"),
      system_prompt="test",
    )
    iterator = _ManualAsyncIterator()
    runner._query_iter = iterator
    runtime = ChatRuntime(
      system_prompt="test",
      build_runner=lambda _event_log, _sid: runner,
      disconnect_handler=runner.on_disconnect,
    )

    await runtime.on_disconnect()

    assert runner._query_iter is None
    assert iterator.closed is True
    assert iterator.close_calls == 1

  _run(case())


def test_on_disconnect_idempotent() -> None:
  calls: list[str] = []

  async def _handler() -> None:
    calls.append("disconnect")

  runtime = ChatRuntime(
    system_prompt="test",
    build_runner=lambda _event_log, _sid: None,
    disconnect_handler=_handler,
  )

  _run(runtime.on_disconnect())
  _run(runtime.on_disconnect())

  assert calls == ["disconnect"]


def test_client_disconnect_releases_lock_fast(make_test_app) -> None:
  async def case() -> None:
    provider = _HangingProvider()
    app = make_test_app(provider=provider)

    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
      session_info = await _init_session(client)
      session = app.state.auth.session_store.get_session(session_info["session_id"])
      assert session is not None

      async with client.stream(
        "POST",
        "/api/chat",
        headers={"Authorization": f"Bearer {session_info['session_token']}"},
        json=_chat_payload(),
      ) as response:
        assert response.status_code == 200
        await _read_sse_event(response)

      elapsed = await _wait_for(lambda: session.stream_active is False, timeout=2.0)

      assert elapsed < 2.0

  _run(case())


def test_disconnect_works_with_externally_built_runtime_no_handler(make_test_app) -> None:
  async def case() -> None:
    provider = _HangingProvider()
    app = make_test_app(
      provider=provider,
      runner_class=_ObservedAgentRunner,
      attach_disconnect_handler=False,
    )

    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
      session_info = await _init_session(client)
      session = app.state.auth.session_store.get_session(session_info["session_id"])
      assert session is not None

      async with client.stream(
        "POST",
        "/api/chat",
        headers={"Authorization": f"Bearer {session_info['session_token']}"},
        json=_chat_payload(),
      ) as response:
        assert response.status_code == 200
        await _read_sse_event(response)
        runtime = app.state.test_state.runtime
        assert runtime is not None
        assert callable(runtime.disconnect_handler)

      elapsed = await _wait_for(
        lambda: session.stream_active is False
        and provider.close_calls >= 1
        and getattr(app.state.test_state.runner, "cancelled_calls", 0) == 1
        and getattr(app.state.test_state.runner, "_disconnected", False) is True,
        timeout=2.0,
      )

      assert elapsed < 2.0
      assert app.state.test_state.disconnect_hook_calls == 0

  _run(case())


def test_disconnect_releases_lock_with_real_uvicorn(make_test_app) -> None:
  async def case() -> None:
    uvicorn = pytest.importorskip("uvicorn")
    app = make_test_app(
      provider=_HangingProvider(),
      runner_class=_ObservedAgentRunner,
      attach_disconnect_handler=False,
    )
    try:
      port = _find_free_port()
    except PermissionError as exc:
      pytest.skip(f"loopback sockets are unavailable in this environment: {exc}")
    config = uvicorn.Config(
      app,
      host="127.0.0.1",
      port=port,
      log_level="error",
      access_log=False,
    )
    server = uvicorn.Server(config)
    server.install_signal_handlers = lambda: None
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    try:
      await _wait_for(lambda: bool(server.started), timeout=5.0, interval=0.01)

      async with httpx.AsyncClient(base_url=f"http://127.0.0.1:{port}", timeout=5.0) as client:
        session_info = await _init_session(client)

        async with client.stream(
          "POST",
          "/api/chat",
          headers={"Authorization": f"Bearer {session_info['session_token']}"},
          json=_chat_payload(),
        ) as response:
          assert response.status_code == 200
          await _read_sse_event(response)

        started = time.perf_counter()
        while True:
          async with client.stream(
            "POST",
            "/api/chat",
            headers={"Authorization": f"Bearer {session_info['session_token']}"},
            json=_chat_payload(),
          ) as response:
            if response.status_code == 200:
              break
            assert response.status_code == 409
          elapsed = time.perf_counter() - started
          if elapsed >= 2.0:
            raise AssertionError("Expected reconnect to succeed within 2s against real uvicorn")
          await asyncio.sleep(0.05)

        assert time.perf_counter() - started < 2.0
        runtime = app.state.test_state.runtime
        assert runtime is not None
        assert callable(runtime.disconnect_handler)
        assert app.state.test_state.disconnect_hook_calls == 0

    finally:
      server.should_exit = True
      thread.join(timeout=5.0)
      if thread.is_alive():
        server.force_exit = True
        thread.join(timeout=5.0)
      assert not thread.is_alive()

  _run(case())


def test_second_request_succeeds_after_disconnect(make_test_app) -> None:
  async def case() -> None:
    app = make_test_app(provider=_HangingProvider())

    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
      session_info = await _init_session(client)
      session = app.state.auth.session_store.get_session(session_info["session_id"])
      assert session is not None

      async with client.stream(
        "POST",
        "/api/chat",
        headers={"Authorization": f"Bearer {session_info['session_token']}"},
        json=_chat_payload(),
      ) as response:
        assert response.status_code == 200
        await _read_sse_event(response)

      await _wait_for(lambda: session.stream_active is False, timeout=2.0)

      started = time.perf_counter()
      async with client.stream(
        "POST",
        "/api/chat",
        headers={"Authorization": f"Bearer {session_info['session_token']}"},
        json=_chat_payload(),
      ) as response:
        assert response.status_code == 200
      assert time.perf_counter() - started < 3.0

  _run(case())


def test_concurrent_request_still_gets_409(make_test_app) -> None:
  async def case() -> None:
    app = make_test_app(provider=_HangingProvider())

    async with (
      httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as first_client,
      httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as second_client,
    ):
      session_info = await _init_session(first_client)
      session = app.state.auth.session_store.get_session(session_info["session_id"])
      assert session is not None

      async with first_client.stream(
        "POST",
        "/api/chat",
        headers={"Authorization": f"Bearer {session_info['session_token']}"},
        json=_chat_payload(),
      ) as response:
        assert response.status_code == 200
        await _read_sse_event(response)

        conflict = await second_client.post(
          "/api/chat",
          headers={"Authorization": f"Bearer {session_info['session_token']}"},
          json=_chat_payload(),
        )
        assert conflict.status_code == 409

      await _wait_for(lambda: session.stream_active is False, timeout=2.0)

  _run(case())


def test_disconnect_does_not_emit_stream_retry(make_test_app) -> None:
  async def case() -> None:
    app = make_test_app(provider=_HangingProvider())

    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
      session_info = await _init_session(client)
      session = app.state.auth.session_store.get_session(session_info["session_id"])
      assert session is not None

      async with client.stream(
        "POST",
        "/api/chat",
        headers={"Authorization": f"Bearer {session_info['session_token']}"},
        json=_chat_payload(),
      ) as response:
        assert response.status_code == 200
        await _read_sse_event(response)

      await _wait_for(lambda: session.stream_active is False, timeout=2.0)

      event_types = [entry.event.get("type") for entry in app.state.test_state.event_log.entries]
      assert "stream_retry" not in event_types

  _run(case())


def test_normal_stream_completion_unchanged(make_test_app) -> None:
  async def case() -> None:
    app = make_test_app(provider=_CompletingProvider(text="done"))

    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
      session_info = await _init_session(client)

      async with client.stream(
        "POST",
        "/api/chat",
        headers={"Authorization": f"Bearer {session_info['session_token']}"},
        json=_chat_payload(),
      ) as response:
        assert response.status_code == 200
        events = await _collect_sse_events(response)

      texts = "".join(event.get("text", "") for event in events if event.get("type") == "text_delta")
      event_types = [event.get("type") for event in events]
      assert texts == "done"
      assert "stream_complete" in event_types
      assert "stream_retry" not in event_types

  _run(case())


def test_normal_completion_leaves_no_pending_disconnect_task(make_test_app, monkeypatch: pytest.MonkeyPatch) -> None:
  async def case() -> None:
    disconnect_tasks: list[asyncio.Task[Any]] = []
    original_create_task = gateway_server.asyncio.create_task

    def _record_create_task(coro, *args: Any, **kwargs: Any):
      task = original_create_task(coro, *args, **kwargs)
      coro_name = getattr(getattr(coro, "cr_code", None), "co_name", "")
      if coro_name == "_safe_fire_disconnect":
        disconnect_tasks.append(task)
      return task

    monkeypatch.setattr(gateway_server.asyncio, "create_task", _record_create_task)
    app = make_test_app(provider=_CompletingProvider())

    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
      session_info = await _init_session(client)

      async with client.stream(
        "POST",
        "/api/chat",
        headers={"Authorization": f"Bearer {session_info['session_token']}"},
        json=_chat_payload(),
      ) as response:
        assert response.status_code == 200
        await _collect_sse_events(response)

      await asyncio.sleep(0)

      assert disconnect_tasks
      assert all(task.done() for task in disconnect_tasks)

  _run(case())


def test_sdk_runner_client_disconnect_releases_lock_fast(make_test_app, monkeypatch: pytest.MonkeyPatch) -> None:
  async def case() -> None:
    iterators: list[_HangingQueryIterator] = []

    def _iterator_factory(_prompt: str, _options: Any):
      iterator = _HangingQueryIterator()
      iterators.append(iterator)
      return iterator

    _install_fake_agent_sdk(monkeypatch, _iterator_factory)
    app = make_test_app(
      provider=None,
      runner_class=AgentSDKRunner,
      sdk_config=AgentSDKConfig(api_key="test-key", model="claude-sonnet-4-6"),
    )

    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
      session_info = await _init_session(client)
      session = app.state.auth.session_store.get_session(session_info["session_id"])
      assert session is not None

      async with client.stream(
        "POST",
        "/api/chat",
        headers={"Authorization": f"Bearer {session_info['session_token']}"},
        json=_chat_payload(),
      ) as response:
        assert response.status_code == 200
        event = await _read_sse_event(response)
        assert event["type"] == "text_delta"

      elapsed = await _wait_for(lambda: session.stream_active is False, timeout=2.0)

      assert elapsed < 2.0
      assert iterators and iterators[0].close_calls >= 1

  _run(case())


def test_sdk_runner_second_request_succeeds_after_disconnect(make_test_app, monkeypatch: pytest.MonkeyPatch) -> None:
  async def case() -> None:
    def _iterator_factory(_prompt: str, _options: Any):
      return _HangingQueryIterator()

    _install_fake_agent_sdk(monkeypatch, _iterator_factory)
    app = make_test_app(
      provider=None,
      runner_class=AgentSDKRunner,
      sdk_config=AgentSDKConfig(api_key="test-key", model="claude-sonnet-4-6"),
    )

    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
      session_info = await _init_session(client)
      session = app.state.auth.session_store.get_session(session_info["session_id"])
      assert session is not None

      async with client.stream(
        "POST",
        "/api/chat",
        headers={"Authorization": f"Bearer {session_info['session_token']}"},
        json=_chat_payload(),
      ) as response:
        assert response.status_code == 200
        await _read_sse_event(response)

      await _wait_for(lambda: session.stream_active is False, timeout=2.0)

      started = time.perf_counter()
      async with client.stream(
        "POST",
        "/api/chat",
        headers={"Authorization": f"Bearer {session_info['session_token']}"},
        json=_chat_payload(),
      ) as response:
        assert response.status_code == 200
      assert time.perf_counter() - started < 3.0

  _run(case())


def test_tool_call_in_flight_waits_for_timeout_short(make_test_app) -> None:
  async def case() -> None:
    dispatcher = _HangingDispatcher(cancel_delay=2.0)
    app = make_test_app(
      provider=_ToolCallProvider(),
      dispatcher=dispatcher,
      tool_definitions=[
        {
          "name": "hang_tool",
          "description": "Hang until cancelled",
          "input_schema": {"type": "object", "properties": {}},
        }
      ],
      tool_call_timeout=2.0,
    )

    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
      session_info = await _init_session(client)
      session = app.state.auth.session_store.get_session(session_info["session_id"])
      assert session is not None

      stream = client.stream(
        "POST",
        "/api/chat",
        headers={"Authorization": f"Bearer {session_info['session_token']}"},
        json=_chat_payload(),
      )
      response = await stream.__aenter__()
      assert response.status_code == 200
      event = await _read_until_event_type(response, "tool_call_start")
      assert event["type"] == "tool_call_start"

      disconnect_started = time.perf_counter()
      await stream.__aexit__(None, None, None)
      elapsed = time.perf_counter() - disconnect_started

      assert dispatcher.cancelled_calls == 1
      assert elapsed >= 1.5
      assert elapsed < 4.0
      assert session.stream_active is False

  _run(case())
