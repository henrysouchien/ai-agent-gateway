from __future__ import annotations

import asyncio
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, AsyncIterator

from ..fixture_gate import (
  FIXTURE_APPROVAL_CANVAS_ARTIFACT_SKILL_NAME,
  FIXTURE_CANVAS_ARTIFACT_SKILL_NAME,
  FIXTURE_DASHBOARD_ARTIFACT_SKILL_NAME,
  FIXTURE_APPROVAL_TOOL_NAME,
  FIXTURE_MODEL_ID,
  FIXTURE_TERMINAL_FAILURE_SKILL_NAME,
)
from ..model_registry import AdapterRouteSupport
from .base import CostEstimate, ModelInfo, ModelProvider, StreamEvent, ThinkingLevel


def _fixture_run_seconds() -> float:
  raw = os.getenv("FIXTURE_RUN_SECONDS", "").strip()
  if not raw:
    raw = os.getenv("AGENT_GATEWAY_FIXTURE_RUN_SECONDS", "").strip()
  if not raw:
    return 5.0
  try:
    return max(0.0, float(raw))
  except ValueError:
    return 5.0


@dataclass
class FixtureClient:
  turn: int = 0


class FixtureProvider(ModelProvider):
  """Deterministic development-only provider for control-surface QA."""

  name = "fixture"

  @classmethod
  def adapter_route_support(cls) -> AdapterRouteSupport:
    # Protocol facts: a deterministic in-process stream with no upstream wire
    # protocol; registry entries binding it must name these exact values.
    return AdapterRouteSupport(
      adapter="fixture.responses",
      provider="fixture",
      protocol_profiles=frozenset({"fixture.deterministic"}),
      routes=frozenset({"fixture.in_process"}),
    )

  def has_active_credential(self, config: dict[str, Any]) -> bool:
    _ = config
    return True

  def create_client(self, config: dict[str, Any], *, timeout: float | None = None) -> FixtureClient:
    _ = config, timeout
    return FixtureClient()

  async def close_client(self, client: Any, timeout: float = 2.0) -> None:
    _ = client, timeout
    return None

  def get_model_info(self, model: str) -> ModelInfo:
    return ModelInfo(
      id=str(model or FIXTURE_MODEL_ID),
      provider=self.name,
      context_window=8_192,
      max_output_tokens=1_024,
      supports_thinking=False,
      supports_vision=False,
      supports_tool_use=True,
      input_cost_per_mtok=0.0,
      output_cost_per_mtok=0.0,
      cache_read_cost_per_mtok=0.0,
      cache_write_cost_per_mtok=0.0,
    )

  def build_request_params(
    self,
    *,
    model: str,
    messages: list[dict[str, Any]],
    system_prompt: str | list[tuple[str, bool]] | None,
    tools: list[dict[str, Any]],
    max_tokens: int,
    thinking_level: ThinkingLevel = ThinkingLevel.HIGH,
    **kwargs: Any,
  ) -> dict[str, Any]:
    params = dict(kwargs)
    params.update(
      {
        "model": model,
        "messages": messages,
        "system_prompt": system_prompt,
        "tools": tools,
        "max_tokens": max_tokens,
        "thinking_level": thinking_level,
      }
    )
    return params

  def normalize_messages(self, messages: list[dict[str, Any]], model_info: ModelInfo) -> list[dict[str, Any]]:
    _ = model_info
    return list(messages)

  async def stream(self, client: Any, params: dict[str, Any]) -> AsyncIterator[StreamEvent]:
    if not isinstance(client, FixtureClient):
      client = FixtureClient()
    client.turn += 1
    requested_skill = _requested_fixture_skill(params)
    if requested_skill == FIXTURE_TERMINAL_FAILURE_SKILL_NAME:
      async for event in self._stream_terminal_failure_turn():
        yield event
      return
    if requested_skill == FIXTURE_DASHBOARD_ARTIFACT_SKILL_NAME:
      if client.turn == 1:
        async for event in self._stream_dashboard_artifact_turn_one():
          yield event
        return
      async for event in self._stream_dashboard_artifact_turn_two(turn=client.turn):
        yield event
      return
    if requested_skill == FIXTURE_APPROVAL_CANVAS_ARTIFACT_SKILL_NAME:
      if client.turn == 1:
        async for event in self._stream_canvas_artifact_turn_one():
          yield event
        return
      if client.turn == 2:
        async for event in self._stream_canvas_artifact_approval_turn(params):
          yield event
        return
      async for event in self._stream_canvas_artifact_approval_complete(turn=client.turn):
        yield event
      return
    if requested_skill == FIXTURE_CANVAS_ARTIFACT_SKILL_NAME:
      if client.turn == 1:
        async for event in self._stream_canvas_artifact_turn_one():
          yield event
        return
      async for event in self._stream_canvas_artifact_turn_two(turn=client.turn):
        yield event
      return
    if client.turn == 1:
      async for event in self._stream_turn_one():
        yield event
      return
    async for event in self._stream_turn_two(params, turn=client.turn):
      yield event

  def estimate_cost(
    self,
    model: str,
    input_tokens: int,
    output_tokens: int,
    cache_read_tokens: int = 0,
    cache_creation_tokens: int = 0,
  ) -> CostEstimate:
    _ = model, input_tokens, output_tokens, cache_read_tokens, cache_creation_tokens
    return CostEstimate()

  async def _stream_turn_one(self) -> AsyncIterator[StreamEvent]:
    text = "Fixture turn 1 running before approval gate.\n"
    yield StreamEvent(type="text_delta", text=text)
    yield StreamEvent(type="text_end", text=text, raw_block={"type": "text", "text": text})
    await asyncio.sleep(_fixture_run_seconds())

    tool_id = "fixture_approval_1"
    tool_input = {
      "reason": "deterministic fixture approval gate",
      "side_effect": "none",
    }
    tool_input_json = json.dumps(tool_input, sort_keys=True, separators=(",", ":"))
    raw_block = {
      "type": "tool_use",
      "id": tool_id,
      "name": FIXTURE_APPROVAL_TOOL_NAME,
      "input": tool_input,
    }
    yield StreamEvent(
      type="tool_use_start",
      tool_id=tool_id,
      tool_name=FIXTURE_APPROVAL_TOOL_NAME,
      raw_block={"type": "tool_use", "id": tool_id, "name": FIXTURE_APPROVAL_TOOL_NAME},
    )
    yield StreamEvent(type="tool_use_delta", tool_input_json=tool_input_json)
    yield StreamEvent(
      type="tool_use_end",
      tool_id=tool_id,
      tool_name=FIXTURE_APPROVAL_TOOL_NAME,
      tool_input_json=tool_input_json,
      tool_input=tool_input,
      raw_block=raw_block,
    )
    yield StreamEvent(type="message_end", stop_reason="tool_use")

  async def _stream_turn_two(self, params: dict[str, Any], *, turn: int) -> AsyncIterator[StreamEvent]:
    steering = _extract_operator_steering(params.get("messages"))
    if steering:
      text = f"Fixture turn {turn} received steering: {steering}\n"
    else:
      text = f"Fixture turn {turn} completed with no injected steering.\n"
    yield StreamEvent(type="text_delta", text=text)
    yield StreamEvent(type="text_end", text=text, raw_block={"type": "text", "text": text})
    await asyncio.sleep(_fixture_run_seconds())
    yield StreamEvent(type="message_end", stop_reason="end_turn")

  async def _stream_canvas_artifact_turn_one(self) -> AsyncIterator[StreamEvent]:
    text = "Fixture Canvas artifact turn 1 emitting deterministic artifact.\n"
    yield StreamEvent(type="text_delta", text=text)
    yield StreamEvent(type="text_end", text=text, raw_block={"type": "text", "text": text})
    await asyncio.sleep(_fixture_run_seconds())

    tool_id = "fixture_canvas_artifact_1"
    tool_input = {
      "title": "Fixture Canvas Artifact",
      "purpose": "exploration",
      "summary": "Deterministic dev-only Canvas artifact fixture for Hank web live QA.",
      "tsx_source": (
        "import React from 'react';\n"
        "import { Canvas, InsightBanner, Prose, SectionHeader } from '@hank/canvas-kit';\n\n"
        "export default function FixtureCanvasArtifact() {\n"
        "  return (\n"
        "    <Canvas title=\"Fixture Canvas Artifact\" generatedAt=\"2026-07-22\">\n"
        "      <InsightBanner title=\"Canvas fixture emitted\" tone=\"positive\">\n"
        "        <Prose>This deterministic artifact verifies the CanvasArtifact live path.</Prose>\n"
        "      </InsightBanner>\n"
        "      <SectionHeader title=\"Status\" description=\"Fixture emitted successfully.\" />\n"
        "    </Canvas>\n"
        "  );\n"
        "}\n"
      ),
      "copy_as_prompt": "Review the deterministic CanvasArtifact fixture output.",
      "copy_as_markdown": (
        "## Fixture Canvas Artifact\n\n"
        "Deterministic dev-only Canvas artifact fixture for Hank web live QA."
      ),
      "copy_as_json": {
        "fixture": "fixture-canvas-artifact",
        "contract_name": "CanvasArtifact",
      },
      "sources": [],
    }
    tool_input_json = json.dumps(tool_input, sort_keys=True, separators=(",", ":"))
    raw_block = {
      "type": "tool_use",
      "id": tool_id,
      "name": "emit_canvas_artifact",
      "input": tool_input,
    }
    yield StreamEvent(
      type="tool_use_start",
      tool_id=tool_id,
      tool_name="emit_canvas_artifact",
      raw_block={"type": "tool_use", "id": tool_id, "name": "emit_canvas_artifact"},
    )
    yield StreamEvent(type="tool_use_delta", tool_input_json=tool_input_json)
    yield StreamEvent(
      type="tool_use_end",
      tool_id=tool_id,
      tool_name="emit_canvas_artifact",
      tool_input_json=tool_input_json,
      tool_input=tool_input,
      raw_block=raw_block,
    )
    yield StreamEvent(type="message_end", stop_reason="tool_use")

  async def _stream_dashboard_artifact_turn_one(self) -> AsyncIterator[StreamEvent]:
    text = "Fixture dashboard artifact turn 1 emitting deterministic artifact.\n"
    yield StreamEvent(type="text_delta", text=text)
    yield StreamEvent(type="text_end", text=text, raw_block={"type": "text", "text": text})
    await asyncio.sleep(_fixture_run_seconds())

    tool_id = "fixture_dashboard_artifact_1"
    tool_input = {
      "payload": _dashboard_fixture_payload(),
      "summary": "Deterministic dev-only DashboardArtifact fixture for Hank web live QA.",
      "profile": "production",
    }
    tool_input_json = json.dumps(tool_input, sort_keys=True, separators=(",", ":"))
    raw_block = {
      "type": "tool_use",
      "id": tool_id,
      "name": "emit_dashboard_artifact",
      "input": tool_input,
    }
    yield StreamEvent(
      type="tool_use_start",
      tool_id=tool_id,
      tool_name="emit_dashboard_artifact",
      raw_block={"type": "tool_use", "id": tool_id, "name": "emit_dashboard_artifact"},
    )
    yield StreamEvent(type="tool_use_delta", tool_input_json=tool_input_json)
    yield StreamEvent(
      type="tool_use_end",
      tool_id=tool_id,
      tool_name="emit_dashboard_artifact",
      tool_input_json=tool_input_json,
      tool_input=tool_input,
      raw_block=raw_block,
    )
    yield StreamEvent(type="message_end", stop_reason="tool_use")

  async def _stream_dashboard_artifact_turn_two(self, *, turn: int) -> AsyncIterator[StreamEvent]:
    text = f"Fixture dashboard artifact turn {turn} completed after artifact emission.\n"
    yield StreamEvent(type="text_delta", text=text)
    yield StreamEvent(type="text_end", text=text, raw_block={"type": "text", "text": text})
    await asyncio.sleep(_fixture_run_seconds())
    yield StreamEvent(type="message_end", stop_reason="end_turn")

  async def _stream_canvas_artifact_turn_two(self, *, turn: int) -> AsyncIterator[StreamEvent]:
    text = f"Fixture Canvas artifact turn {turn} completed after artifact emission.\n"
    yield StreamEvent(type="text_delta", text=text)
    yield StreamEvent(type="text_end", text=text, raw_block={"type": "text", "text": text})
    await asyncio.sleep(_fixture_run_seconds())
    yield StreamEvent(type="message_end", stop_reason="end_turn")

  async def _stream_canvas_artifact_approval_turn(self, params: dict[str, Any]) -> AsyncIterator[StreamEvent]:
    text = "Fixture Canvas artifact turn 2 requesting approval with artifact evidence.\n"
    yield StreamEvent(type="text_delta", text=text)
    yield StreamEvent(type="text_end", text=text, raw_block={"type": "text", "text": text})
    await asyncio.sleep(_fixture_run_seconds())

    artifact_id = _extract_artifact_id(params.get("messages")) or "fixture-canvas-artifact-missing"
    artifact_path = f"artifacts/_canvas/{artifact_id}.json"
    binary_artifact_path = f"artifacts/_canvas/{artifact_id}.bundle.js"
    tool_id = "fixture_canvas_artifact_approval_1"
    tool_input = {
      "reason": "deterministic fixture approval gate with CanvasArtifact evidence",
      "side_effect": "none",
      "evidence_artifact": {
        "artifact_id": artifact_id,
        "title": "Fixture Canvas Approval Evidence",
        "skill": "_canvas",
        "contract_name": "CanvasArtifact",
        "artifact_path": artifact_path,
        "binary_artifact_path": binary_artifact_path,
        "data_source": "fixture",
      },
    }
    tool_input_json = json.dumps(tool_input, sort_keys=True, separators=(",", ":"))
    raw_block = {
      "type": "tool_use",
      "id": tool_id,
      "name": FIXTURE_APPROVAL_TOOL_NAME,
      "input": tool_input,
    }
    yield StreamEvent(
      type="tool_use_start",
      tool_id=tool_id,
      tool_name=FIXTURE_APPROVAL_TOOL_NAME,
      raw_block={"type": "tool_use", "id": tool_id, "name": FIXTURE_APPROVAL_TOOL_NAME},
    )
    yield StreamEvent(type="tool_use_delta", tool_input_json=tool_input_json)
    yield StreamEvent(
      type="tool_use_end",
      tool_id=tool_id,
      tool_name=FIXTURE_APPROVAL_TOOL_NAME,
      tool_input_json=tool_input_json,
      tool_input=tool_input,
      raw_block=raw_block,
    )
    yield StreamEvent(type="message_end", stop_reason="tool_use")

  async def _stream_canvas_artifact_approval_complete(self, *, turn: int) -> AsyncIterator[StreamEvent]:
    text = f"Fixture Canvas artifact approval turn {turn} completed after approval.\n"
    yield StreamEvent(type="text_delta", text=text)
    yield StreamEvent(type="text_end", text=text, raw_block={"type": "text", "text": text})
    await asyncio.sleep(_fixture_run_seconds())
    yield StreamEvent(type="message_end", stop_reason="end_turn")

  async def _stream_terminal_failure_turn(self) -> AsyncIterator[StreamEvent]:
    text = "Fixture terminal failure run intentionally failing for Agent Control QA.\n"
    yield StreamEvent(type="text_delta", text=text)
    yield StreamEvent(type="text_end", text=text, raw_block={"type": "text", "text": text})
    await asyncio.sleep(_fixture_run_seconds())
    raise RuntimeError("fixture_terminal_failure: deterministic failure for Agent Control QA")


def _extract_operator_steering(messages: Any) -> str:
  if not isinstance(messages, list):
    return ""
  for message in reversed(messages):
    if not isinstance(message, dict):
      continue
    if message.get("role") != "user":
      continue
    content = message.get("content")
    if not isinstance(content, str):
      continue
    if "Operator update for this task" not in content:
      continue
    lines = [line.strip() for line in content.splitlines()]
    for line in reversed(lines):
      if line.startswith("- id=") and ":" in line:
        return line.split(":", 1)[1].strip()
    return content.strip()
  return ""


def _extract_artifact_id(messages: Any) -> str:
  if not isinstance(messages, list):
    return ""
  for message in reversed(messages):
    if not isinstance(message, dict):
      continue
    content = message.get("content")
    if not isinstance(content, list):
      continue
    for block in reversed(content):
      if not isinstance(block, dict) or block.get("type") != "tool_result":
        continue
      raw_content = block.get("content")
      if not isinstance(raw_content, str) or not raw_content.strip():
        continue
      try:
        payload = json.loads(raw_content)
      except json.JSONDecodeError:
        continue
      if not isinstance(payload, dict):
        continue
      artifact_id = str(payload.get("artifact_id") or "").strip()
      if artifact_id:
        return artifact_id
  return ""


def _requested_fixture_skill(params: dict[str, Any]) -> str:
  haystack: list[str] = []
  system_prompt = params.get("system_prompt")
  if isinstance(system_prompt, str):
    haystack.append(system_prompt)
  elif isinstance(system_prompt, list):
    haystack.extend(str(entry[0]) for entry in system_prompt if isinstance(entry, tuple) and entry)
  messages = params.get("messages")
  if isinstance(messages, list):
    for message in messages:
      if not isinstance(message, dict):
        continue
      content = message.get("content")
      if isinstance(content, str):
        haystack.append(content)
  combined = "\n".join(haystack)
  if FIXTURE_DASHBOARD_ARTIFACT_SKILL_NAME in combined:
    return FIXTURE_DASHBOARD_ARTIFACT_SKILL_NAME
  if FIXTURE_APPROVAL_CANVAS_ARTIFACT_SKILL_NAME in combined:
    return FIXTURE_APPROVAL_CANVAS_ARTIFACT_SKILL_NAME
  if FIXTURE_CANVAS_ARTIFACT_SKILL_NAME in combined:
    return FIXTURE_CANVAS_ARTIFACT_SKILL_NAME
  if FIXTURE_TERMINAL_FAILURE_SKILL_NAME in combined:
    return FIXTURE_TERMINAL_FAILURE_SKILL_NAME
  return ""


def _dashboard_fixture_payload() -> dict[str, Any]:
  for parent in Path(__file__).resolve().parents:
    path = parent / "tests" / "fixtures" / "dashboard" / "full.payload.json"
    if path.is_file():
      return json.loads(path.read_text(encoding="utf-8"))
  raise FileNotFoundError("tests/fixtures/dashboard/full.payload.json")


__all__ = ["FixtureClient", "FixtureProvider"]
