import sys
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[3]
PKG_DIR = ROOT / "packages" / "agent-gateway"
if str(PKG_DIR) not in sys.path:
  sys.path.insert(0, str(PKG_DIR))

from agent_gateway.server import ChatRuntime, GatewayServerConfig, create_gateway_app


class _CompleteRunner:
  def __init__(self, event_log, calls: list[dict[str, Any]]) -> None:
    self._event_log = event_log
    self._calls = calls

  async def run(
    self,
    *,
    messages: list[dict[str, Any]],
    system_prompt: str | None = None,
    model_override: str | None = None,
    max_turns: int | None = None,
  ) -> None:
    self._calls.append(
      {
        "messages": messages,
        "system_prompt": system_prompt,
        "model_override": model_override,
        "max_turns": max_turns,
      }
    )
    self._event_log.append({"type": "stream_complete", "usage": {}})


def _make_app(*, resolved_provider_name: str, model_override: str):
  calls: list[dict[str, Any]] = []

  async def _build_chat_runtime(session, request, channel, auth_manager):
    _ = session, request, channel, auth_manager
    return ChatRuntime(
      system_prompt="system",
      build_runner=lambda event_log, _sid: _CompleteRunner(event_log, calls),
      model_override=model_override,
      resolved_provider_name=resolved_provider_name,
    )

  app = create_gateway_app(
    GatewayServerConfig(
      auth_config={"model": "claude-opus-4-7"},
      allowed_models={"claude-opus-4-7"},
      build_chat_runtime=_build_chat_runtime,
    )
  )
  return app, calls


def _init(client: TestClient) -> str:
  response = client.post("/api/chat/init", json={"api_key": "gateway-key", "user_id": "alice"})
  assert response.status_code == 200, response.text
  return response.json()["session_token"]


def _chat(client: TestClient, token: str):
  return client.stream(
    "POST",
    "/api/chat",
    headers={"Authorization": f"Bearer {token}"},
    json={"messages": [{"role": "user", "content": "hi"}]},
  )


def test_server_validates_model_against_resolved_provider_allowlist() -> None:
  app, calls = _make_app(resolved_provider_name="openai", model_override="gpt-5.5")
  client = TestClient(app)
  token = _init(client)

  with _chat(client, token) as response:
    assert response.status_code == 200, response.text
    list(response.iter_lines())

  assert calls[0]["model_override"] == "gpt-5.5"


def test_server_rejects_model_not_in_resolved_provider_allowlist() -> None:
  app, _calls = _make_app(resolved_provider_name="anthropic", model_override="gpt-5.5")
  client = TestClient(app)
  token = _init(client)

  with _chat(client, token) as response:
    assert response.status_code == 200
    body = "\n".join(response.iter_lines())
    assert "Invalid model: gpt-5.5" in body
