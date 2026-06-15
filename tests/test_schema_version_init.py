from __future__ import annotations

import json
from typing import Any

from fastapi.testclient import TestClient

from agent_gateway.server import ChatRuntime, GatewayServerConfig, create_gateway_app


class _StreamingRunner:
  def __init__(self, event_log: Any) -> None:
    self._event_log = event_log

  async def run(self, **_: Any) -> None:
    self._event_log.append({"type": "text_delta", "text": "hello", "future_only": "strip-me"})
    self._event_log.append({"type": "future_event", "value": 1})
    self._event_log.append(
      {
        "type": "tool_output_chunk",
        "tool_call_id": "toolu_1",
        "tool_name": "code_execute",
        "stream": "stdout",
        "text": "line",
        "seq": 5,
      }
    )
    self._event_log.append({"type": "stream_complete", "usage": {}})


def _make_app():
  async def _build_chat_runtime(session, request, channel, auth_manager):
    _ = (session, request, channel, auth_manager)
    return ChatRuntime(
      system_prompt="test",
      build_runner=lambda event_log, *_args: _StreamingRunner(event_log),
    )

  return create_gateway_app(
    GatewayServerConfig(
      auth_config={"model": "claude-sonnet-4-6"},
      build_chat_runtime=_build_chat_runtime,
    )
  )


def _init(client: TestClient, payload: dict[str, Any] | None = None) -> dict[str, Any]:
  response = client.post(
    "/api/chat/init",
    json={"api_key": "gateway-key", "user_id": "alice", **(payload or {})},
  )
  assert response.status_code == 200, response.text
  return response.json()


def _sse_payloads(response_text: str) -> list[dict[str, Any]]:
  payloads: list[dict[str, Any]] = []
  for line in response_text.splitlines():
    if line.startswith("data: "):
      payloads.append(json.loads(line[6:]))
  return payloads


def _without_product_id(event: dict[str, Any]) -> dict[str, Any]:
  return {key: value for key, value in event.items() if key != "product_id"}


def test_chat_init_defaults_schema_version_and_echoes_response() -> None:
  app = _make_app()

  with TestClient(app) as client:
    init = _init(client)

  session = app.state.auth.session_store.get_session(init["session_id"])
  assert init["schema_version"] == 1
  assert session.schema_version == 1


def test_chat_init_accepts_explicit_v1() -> None:
  app = _make_app()

  with TestClient(app) as client:
    init = _init(client, {"schema_version": 1})

  session = app.state.auth.session_store.get_session(init["session_id"])
  assert init["schema_version"] == 1
  assert session.schema_version == 1


def test_chat_init_rejects_unsupported_schema_version() -> None:
  app = _make_app()

  with TestClient(app) as client:
    response = client.post(
      "/api/chat/init",
      json={"api_key": "gateway-key", "user_id": "alice", "schema_version": 99},
    )

  assert response.status_code == 400
  assert response.json()["error"] == "unsupported_schema_version"
  assert "supported: [1]" in response.json()["message"]


def test_subscribe_rejects_schema_version_query_param() -> None:
  app = _make_app()

  with TestClient(app) as client:
    init = _init(client)
    response = client.get(
      f"/api/chat/subscribe?session_id={init['session_id']}&schema_version=1",
      headers={"Authorization": f"Bearer {init['session_token']}"},
    )

  assert response.status_code == 400
  assert "schema_version is set at session init" in response.json()["detail"]


def test_stream_envelopes_echo_session_schema_and_apply_v1_projection() -> None:
  app = _make_app()

  with TestClient(app) as client:
    init = _init(client, {"schema_version": 1})
    response = client.post(
      "/api/chat",
      json={"messages": [{"role": "user", "content": "hello"}], "context": {}},
      headers={"Authorization": f"Bearer {init['session_token']}"},
    )

  assert response.status_code == 200, response.text
  payloads = _sse_payloads(response.text)
  assert [payload["schema_version"] for payload in payloads] == [1, 1, 1]
  assert [payload["seq"] for payload in payloads] == [1, 3, 4]
  assert _without_product_id(payloads[0]["event"]) == {"type": "text_delta", "text": "hello"}
  assert _without_product_id(payloads[1]["event"]) == {
    "type": "tool_output_chunk",
    "tool_call_id": "toolu_1",
    "tool_name": "code_execute",
    "stream": "stdout",
    "text": "line",
    "seq": 5,
  }
  assert _without_product_id(payloads[2]["event"]) == {"type": "stream_complete", "usage": {}}
