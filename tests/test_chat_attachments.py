from __future__ import annotations

import base64
import hashlib

import pytest
from agent_gateway import server_models
from agent_gateway.model_registry import (
  INITIAL_MODEL_REGISTRY,
  INITIAL_MODEL_SELECTION_POLICY,
)
from agent_gateway.server import GatewayServerConfig, create_gateway_app
from agent_gateway.server_models import ChatRequest
from fastapi.testclient import TestClient
from pydantic import ValidationError


def _wire(text: str = "FY2026 selected facts\n") -> dict[str, object]:
  payload = text.encode("utf-8")
  return {
    "schema_version": "chat-attachment/1",
    "input_name": "source_document",
    "display_name": "facts.txt",
    "media_type": "text/plain",
    "encoding": "utf-8",
    "content_bytes": len(payload),
    "content_sha256": hashlib.sha256(payload).hexdigest(),
    "content_b64": base64.b64encode(payload).decode("ascii"),
  }


def test_risk_wire_shape_round_trips_exact_bytes() -> None:
  request = ChatRequest.model_validate({
    "messages": [{"role": "user", "content": "Use the selected content."}],
    "attachments": [_wire("café 🚀\n")],
  })

  assert request.attachments[0].decoded_bytes() == "café 🚀\n".encode("utf-8")
  assert request.model_dump(mode="json")["attachments"][0] == _wire("café 🚀\n")


def test_ai_decodes_attachment_base64_once(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  decode = server_models.base64.b64decode
  calls = 0

  def _counting_decode(*args, **kwargs):
    nonlocal calls
    calls += 1
    return decode(*args, **kwargs)

  monkeypatch.setattr(server_models.base64, "b64decode", _counting_decode)
  request = ChatRequest.model_validate({
    "messages": [{"role": "user", "content": "Use it."}],
    "attachments": [_wire("exact once")],
  })

  assert calls == 1
  assert request.attachments[0].decoded_bytes() == b"exact once"
  assert calls == 1


def test_requests_without_attachments_keep_empty_tuple() -> None:
  request = ChatRequest(messages=[{"role": "user", "content": "hello"}])

  assert request.attachments == ()


def test_investment_selection_is_one_exact_optional_coordinate() -> None:
  request = ChatRequest.model_validate({
    "messages": [{"role": "user", "content": "Use it."}],
    "investment_artifact_selection": {
      "artifact_id": "artifact:quant-1",
      "view": "summary",
    },
  })

  assert request.attachments == ()
  assert request.model_dump(mode="json")["investment_artifact_selection"] == {
    "artifact_id": "artifact:quant-1",
    "view": "summary",
  }


@pytest.mark.parametrize(
  "selection",
  [
    {"artifact_id": "artifact:quant-1", "view": "schema"},
    {"artifact_id": " artifact:quant-1", "view": "summary"},
    {"artifact_id": "artifact:quant-1", "view": "SUMMARY"},
    {"artifact_id": "artifact:quant-1", "view": "summary", "run_id": "run-1"},
  ],
)
def test_investment_selection_rejects_non_exact_or_extra_coordinates(
  selection: dict[str, object],
) -> None:
  with pytest.raises(ValidationError):
    ChatRequest.model_validate({
      "messages": [{"role": "user", "content": "Use it."}],
      "investment_artifact_selection": selection,
    })


@pytest.mark.parametrize(
  ("field", "value"),
  (
    ("content_b64", "not-base64"),
    ("content_sha256", "0" * 64),
    ("content_bytes", 999),
    ("media_type", "application/octet-stream"),
    ("display_name", "../facts.txt"),
  ),
)
def test_malformed_attachment_values_fail_closed(field: str, value: object) -> None:
  wire = _wire()
  wire[field] = value

  with pytest.raises(ValidationError):
    ChatRequest.model_validate({
      "messages": [{"role": "user", "content": "Use it."}],
      "attachments": [wire],
    })


def test_non_utf8_and_noncanonical_wire_fail_closed() -> None:
  binary = b"\xff\xfe"
  wire = _wire()
  wire.update({
    "content_bytes": len(binary),
    "content_sha256": hashlib.sha256(binary).hexdigest(),
    "content_b64": base64.b64encode(binary).decode("ascii"),
  })
  with pytest.raises(ValidationError):
    ChatRequest.model_validate({
      "messages": [{"role": "user", "content": "Use it."}],
      "attachments": [wire],
    })

  misplaced = _wire()
  misplaced["input_name"] = "source_document_2"
  with pytest.raises(ValidationError):
    ChatRequest.model_validate({
      "messages": [{"role": "user", "content": "Use it."}],
      "attachments": [misplaced],
    })


def test_attachment_count_item_and_turn_limits_fail_closed() -> None:
  too_many = []
  for index in range(1, 10):
    wire = _wire(str(index))
    wire["input_name"] = (
      "source_document" if index == 1 else f"source_document_{index}"
    )
    too_many.append(wire)
  with pytest.raises(ValidationError, match="more than 8"):
    ChatRequest.model_validate({
      "messages": [{"role": "user", "content": "Use them."}],
      "attachments": too_many,
    })

  oversized_item = _wire()
  oversized_item["content_bytes"] = 1024 * 1024 + 1
  with pytest.raises(ValidationError):
    ChatRequest.model_validate({
      "messages": [{"role": "user", "content": "Use it."}],
      "attachments": [oversized_item],
    })

  aggregate = []
  for index in range(1, 6):
    payload = bytes([64 + index]) * (900 * 1024)
    aggregate.append({
      "schema_version": "chat-attachment/1",
      "input_name": (
        "source_document" if index == 1 else f"source_document_{index}"
      ),
      "display_name": f"facts-{index}.txt",
      "media_type": "text/plain",
      "encoding": "utf-8",
      "content_bytes": len(payload),
      "content_sha256": hashlib.sha256(payload).hexdigest(),
      "content_b64": base64.b64encode(payload).decode("ascii"),
    })
  with pytest.raises(ValidationError, match="aggregate decoded byte limit"):
    ChatRequest.model_validate({
      "messages": [{"role": "user", "content": "Use them."}],
      "attachments": aggregate,
    })


def test_gateway_validation_error_does_not_reflect_attachment_bytes() -> None:
  async def _unused_runtime(*_args, **_kwargs):
    raise AssertionError("invalid requests must not reach runtime construction")

  sensitive_text = "PRIVATE-ATTACHMENT-BYTES-DO-NOT-REFLECT"
  wire = _wire(sensitive_text)
  wire["content_sha256"] = "0" * 64
  encoded = str(wire["content_b64"])
  app = create_gateway_app(GatewayServerConfig(
    tenant_id="test-product",
    model_registry=INITIAL_MODEL_REGISTRY,
    model_selection_policy=INITIAL_MODEL_SELECTION_POLICY,
    build_chat_runtime=_unused_runtime,
  ))

  with TestClient(app) as client:
    response = client.post("/api/chat", json={
      "messages": [{"role": "user", "content": "Use it."}],
      "attachments": [wire],
    })

  assert response.status_code == 422
  assert sensitive_text not in response.text
  assert encoded not in response.text
