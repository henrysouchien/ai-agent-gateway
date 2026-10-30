from __future__ import annotations

import asyncio
import json
from typing import Any

import httpx
import pytest

from hank_model_transport.providers.anthropic import _rejection_error_code
from hank_model_transport.providers.codex_helpers import _ResponsesStreamState
from hank_model_transport.providers.copilot import _map_copilot_event
from hank_model_transport.providers.muse import _only_usage_limit
from hank_model_transport.providers.ollama import OllamaLocalProvider
from hank_model_transport.providers.openai_responses_helpers import (
  _ResponsesStreamState as OpenAIResponsesStreamState,
)
from hank_model_transport.providers.openai_responses_helpers import map_event


@pytest.mark.parametrize("code", [42, False, ["oauth_not_allowed_for_organization"]])
def test_anthropic_rejection_rejects_non_string_error_code(code: Any) -> None:
  exc = RuntimeError("rejected")
  exc.body = {"error": {"type": "permission_error", "details": {"error_code": code}}}  # type: ignore[attr-defined]

  with pytest.raises(TypeError, match="error code must be a string"):
    _rejection_error_code(exc)


@pytest.mark.parametrize("code", [42, False, ["rate_limit_exceeded"]])
def test_copilot_event_rejects_non_string_code_before_classification(code: Any) -> None:
  with pytest.raises(TypeError, match="provider error code must be a string"):
    _map_copilot_event(
      {"type": "error", "message": "bad request", "code": code},
      _ResponsesStreamState(),
    )


@pytest.mark.parametrize(
  "event",
  [
    {"type": "error", "message": "bad request", "code": False},
    {"type": "error", "message": "bad request", "code": ["rate_limit_exceeded"]},
    {
      "type": "response.failed",
      "response": {"error": {"message": "failed", "code": 42}},
    },
  ],
)
def test_openai_event_rejects_non_string_code_before_classification(event: dict[str, Any]) -> None:
  with pytest.raises(TypeError, match="provider error code must be a string"):
    map_event(event, OpenAIResponsesStreamState())


def _muse_error(primary_code: Any) -> RuntimeError:
  exc = RuntimeError(json.dumps({
    "error": {"message": "bad request", "type": "other_error", "code": primary_code},
  }))
  exc.error_code = "rate_limit_exceeded"  # type: ignore[attr-defined]
  return exc


@pytest.mark.parametrize("code", [None, ""])
def test_muse_present_falsy_primary_code_does_not_fall_back_to_exception_code(code: Any) -> None:
  assert _only_usage_limit(_muse_error(code)) is False


@pytest.mark.parametrize("code", [False, 0, 42, ["rate_limit_exceeded"]])
def test_muse_wrong_typed_primary_code_surfaces_malformed_input(code: Any) -> None:
  with pytest.raises(TypeError, match="Muse error code must be a string"):
    _only_usage_limit(_muse_error(code))


def test_ollama_stream_rejects_wrong_typed_code_instead_of_type_fallback() -> None:
  def handle(_request: httpx.Request) -> httpx.Response:
    event = {"error": {"message": "bad request", "code": 42, "type": "rate_limit_error"}}
    return httpx.Response(
      200,
      headers={"content-type": "text/event-stream"},
      content=f"data: {json.dumps(event)}\n\n".encode(),
    )

  async def collect() -> None:
    provider = OllamaLocalProvider()
    client = provider.create_client({"api_key": "", "_transport": httpx.MockTransport(handle)})
    try:
      with pytest.raises(TypeError, match="provider error code must be a string"):
        async for _event in provider.stream(client, {"model": "llama-fixture"}):
          pass
    finally:
      await provider.close_client(client)

  asyncio.run(collect())
