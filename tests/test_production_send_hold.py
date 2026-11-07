"""agent_gateway's own Telegram senders send only from a production runtime.

The guard is ``agent_gateway.production_identity``: the sender's code location, not the
environment. ``dev_runtime`` puts this process where a checkout or isolated instance runs;
the autouse ``_production_send_authority`` (conftest) is the production default.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import urllib.request
from pathlib import Path
from typing import Any

import pytest

import agent_gateway.autonomous as autonomous
from agent_gateway import production_identity
from agent_gateway.approval_notifications import build_env_telegram_approval_notification_sender


class _RecordingAsyncClient:
  def __init__(self, calls: list[str], *args: Any, **kwargs: Any) -> None:
    _ = args, kwargs
    self._calls = calls

  async def __aenter__(self):
    return self

  async def __aexit__(self, exc_type, exc, tb) -> None:
    _ = exc_type, exc, tb

  async def post(self, url: str, json: dict[str, Any] | None = None, headers: dict[str, str] | None = None):
    _ = json, headers
    self._calls.append(url)
    return _Ok()


class _Ok:
  def raise_for_status(self) -> None:
    return None


class _Urlopen:
  def __init__(self) -> None:
    self.urls: list[str] = []

  def __call__(self, request, timeout=None):
    _ = timeout
    self.urls.append(request.full_url)
    return self

  def __enter__(self):
    return self

  def __exit__(self, *_exc) -> None:
    return None

  def read(self) -> bytes:
    return b"{}"


@pytest.fixture
def telegram_calls(monkeypatch: pytest.MonkeyPatch) -> list[str]:
  calls: list[str] = []
  monkeypatch.setattr(
    autonomous.httpx, "AsyncClient", lambda *args, **kwargs: _RecordingAsyncClient(calls, *args, **kwargs)
  )
  return calls


def test_send_telegram_raises_held_before_any_request_under_dev(
  dev_runtime: Path, telegram_calls: list[str]
) -> None:
  with pytest.raises(production_identity.ProductionSendHeld):
    asyncio.run(autonomous.send_telegram("hello", "bot-token", "chat-id"))
  assert telegram_calls == []


def test_send_telegram_sends_under_production(telegram_calls: list[str]) -> None:
  asyncio.run(autonomous.send_telegram("hello", "bot-token", "chat-id"))
  assert telegram_calls == ["https://api.telegram.org/botbot-token/sendMessage"]


def test_delivery_config_sends_nothing_under_dev_and_reports_the_hold(
  dev_runtime: Path, telegram_calls: list[str], caplog: pytest.LogCaptureFixture, tmp_path: Path
) -> None:
  briefing = tmp_path / "briefing.md"
  briefing.write_text("Findings\n", encoding="utf-8")
  config = autonomous.DeliveryConfig(
    telegram_bot_token="bot-token",
    telegram_chat_id="chat-id",
    telegram_label="Run label",
    briefing_file=briefing,
  )
  with caplog.at_level(logging.WARNING):
    asyncio.run(autonomous.deliver(config, autonomous.RunOutput("Summary", ["tool"], {}, None, False), {}))
  assert telegram_calls == []
  assert "telegram send held: dev_run" in caplog.text
  assert "Telegram delivery failed" not in caplog.text


def test_delivery_config_sends_summary_and_briefing_under_production(
  telegram_calls: list[str], tmp_path: Path
) -> None:
  briefing = tmp_path / "briefing.md"
  briefing.write_text("Findings\n", encoding="utf-8")
  config = autonomous.DeliveryConfig(
    telegram_bot_token="bot-token", telegram_chat_id="chat-id", briefing_file=briefing
  )
  asyncio.run(autonomous.deliver(config, autonomous.RunOutput("Summary", ["tool"], {}, None, False), {}))
  assert len(telegram_calls) == 2


def test_webhook_delivery_is_held_under_dev_and_reported_not_failed(
  dev_runtime: Path, telegram_calls: list[str], caplog: pytest.LogCaptureFixture
) -> None:
  config = autonomous.DeliveryConfig(webhook_url="https://hooks.example.test/run")
  with caplog.at_level(logging.WARNING):
    asyncio.run(autonomous.deliver(config, autonomous.RunOutput("Summary", ["tool"], {}, None, False), {}))
  assert telegram_calls == []
  assert "webhook send held: dev_run" in caplog.text
  assert "Webhook delivery failed" not in caplog.text


def test_webhook_delivery_posts_under_production(telegram_calls: list[str]) -> None:
  config = autonomous.DeliveryConfig(webhook_url="https://hooks.example.test/run")
  asyncio.run(autonomous.deliver(config, autonomous.RunOutput("Summary", ["tool"], {}, None, False), {}))
  assert telegram_calls == ["https://hooks.example.test/run"]


def test_send_webhook_raises_held_before_any_request_under_dev(
  dev_runtime: Path, telegram_calls: list[str]
) -> None:
  with pytest.raises(production_identity.ProductionSendHeld):
    asyncio.run(autonomous.send_webhook("https://hooks.example.test/run", {"a": 1}))
  assert telegram_calls == []


async def _deliver(sender: Any, row: dict[str, str]) -> None:
  result = sender(row)
  assert inspect.isawaitable(result)
  await result


_ROW = {"channel": "telegram", "destination": "1000001", "message": "Approval needed"}
_ENV = {"GATEWAY_APPROVAL_TELEGRAM_BOT_TOKEN": "111111:dummy"}


def test_approval_notification_sender_refuses_before_reading_the_row_under_dev(
  dev_runtime: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
  urlopen = _Urlopen()
  monkeypatch.setattr(urllib.request, "urlopen", urlopen)
  sender = build_env_telegram_approval_notification_sender(_ENV)
  assert sender is not None
  with pytest.raises(production_identity.ProductionSendHeld):
    asyncio.run(_deliver(sender, _ROW))
  assert urlopen.urls == []


def test_approval_notification_sender_sends_under_production(monkeypatch: pytest.MonkeyPatch) -> None:
  urlopen = _Urlopen()
  monkeypatch.setattr(urllib.request, "urlopen", urlopen)
  sender = build_env_telegram_approval_notification_sender(_ENV)
  assert sender is not None
  asyncio.run(_deliver(sender, _ROW))
  assert urlopen.urls == ["https://api.telegram.org/bot111111:dummy/sendMessage"]

