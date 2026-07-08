# ruff: noqa: E402

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
PKG_DIR = ROOT / "packages" / "agent-gateway"
if str(PKG_DIR) not in sys.path:
  sys.path.insert(0, str(PKG_DIR))

import agent_gateway.sub_agent as sub_agent_module
import agent_gateway.sub_agent_skill_state as skill_state


class _Store:
  def __init__(self, initial: dict[str, Any]) -> None:
    self.initial = initial
    self.saved: dict[str, dict[str, Any]] = {}

  def get(self, name: str) -> dict[str, Any]:
    _ = name
    return dict(self.initial)

  def set(self, name: str, value: dict[str, Any]) -> None:
    self.saved[name] = dict(value)


def test_parent_skill_state_wrappers_delegate_to_sidecar(monkeypatch) -> None:
  calls: list[tuple[str, Any]] = []

  def fake_response_text(result: Any | None) -> str:
    calls.append(("response", result))
    return "patched-response"

  def fake_prompt(skill_name: str, previous_state: dict[str, Any]) -> str:
    calls.append(("prompt", (skill_name, previous_state)))
    return "patched-prompt"

  monkeypatch.setattr(skill_state, "result_response_text", fake_response_text)
  monkeypatch.setattr(skill_state, "skill_state_prompt", fake_prompt)

  assert sub_agent_module._result_response_text({"response": "raw"}) == "patched-response"
  assert sub_agent_module._skill_state_prompt("skill-a", {"runs": 1}) == "patched-prompt"
  assert calls == [
    ("response", {"response": "raw"}),
    ("prompt", ("skill-a", {"runs": 1})),
  ]


def test_persist_skill_state_merges_model_state_and_error() -> None:
  async def scenario() -> None:
    store = _Store({"run_count": 2, "keep": "yes", "last_error": {"old": True}})
    profile = SimpleNamespace(name="skill-a", persist_state=True, version="1.2.3")
    warnings: list[tuple[str, tuple[Any, ...]]] = []
    logger = SimpleNamespace(warning=lambda message, *args, **_kwargs: warnings.append((message, args)))

    await skill_state.persist_skill_state(
      {"response": "ignored"},
      {"code": "failed"},
      agent_name="skill-a",
      profile=profile,
      skill_state_store=store,
      skill_state_lock=asyncio.Lock(),
      effective_model="model-a",
      extract_state_update_fn=lambda _text: {"fresh": "state"},
      logger=logger,
    )

    saved = store.saved["skill-a"]
    assert saved["keep"] == "yes"
    assert saved["fresh"] == "state"
    assert saved["model"] == "model-a"
    assert saved["run_count"] == 3
    assert saved["version"] == "1.2.3"
    assert saved["last_error"] == {"code": "failed"}
    assert "last_run" in saved
    assert warnings == []

  asyncio.run(scenario())


def test_persist_skill_state_skips_when_profile_does_not_persist() -> None:
  async def scenario() -> None:
    store = _Store({})
    profile = SimpleNamespace(name="skill-a", persist_state=False, version=None)

    await skill_state.persist_skill_state(
      {"response": "ignored"},
      None,
      agent_name="skill-a",
      profile=profile,
      skill_state_store=store,
      skill_state_lock=asyncio.Lock(),
      effective_model="model-a",
      extract_state_update_fn=lambda _text: {"fresh": "state"},
      logger=SimpleNamespace(warning=lambda *_args, **_kwargs: None),
    )

    assert store.saved == {}

  asyncio.run(scenario())
