from __future__ import annotations

import asyncio
import inspect
import json
import os
import re
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Iterable, Literal, Sequence

from .approval_policy import ApprovalRequest


ApprovalNotificationChannel = Literal["telegram", "email", "push"]
ApprovalNotificationState = Literal[
  "pending",
  "sent",
  "skipped_no_destination",
  "skipped_policy",
  "failed_retryable",
  "failed_terminal",
]

PUBLIC_APPROVAL_NOTIFICATION_CHANNELS = frozenset({"telegram", "email", "push"})


@dataclass(frozen=True, kw_only=True)
class ApprovalNotificationDestination:
  channel: ApprovalNotificationChannel
  destination: str
  verified: bool = True


ApprovalNotificationDestinationResolver = Callable[
  [ApprovalRequest],
  Sequence[ApprovalNotificationDestination | dict[str, Any]]
  | Awaitable[Sequence[ApprovalNotificationDestination | dict[str, Any]]],
]
ApprovalNotificationSender = Callable[[dict[str, Any]], None | Awaitable[None]]


def approval_notification_policy_for_request(request: ApprovalRequest) -> Literal["interrupt", "skip"]:
  if request.notification_policy == "interrupt":
    return "interrupt"
  if request.notification_policy == "skip":
    return "skip"
  if request.tool_class in {"irreversible", "external_write", "portfolio_config"}:
    return "interrupt"
  return "skip"


def normalize_approval_notification_destinations(
  values: Iterable[ApprovalNotificationDestination | dict[str, Any]],
) -> list[ApprovalNotificationDestination]:
  destinations: list[ApprovalNotificationDestination] = []
  seen: set[tuple[str, str]] = set()
  for value in values:
    if isinstance(value, ApprovalNotificationDestination):
      destination = value
    elif isinstance(value, dict):
      channel = str(value.get("channel") or "").strip().lower()
      raw_destination = str(value.get("destination") or value.get("chat_id") or value.get("address") or "").strip()
      destination = ApprovalNotificationDestination(
        channel=channel,  # type: ignore[arg-type]
        destination=raw_destination,
        verified=bool(value.get("verified", True)),
      )
    else:
      continue
    if destination.channel not in PUBLIC_APPROVAL_NOTIFICATION_CHANNELS:
      continue
    if not destination.destination or not destination.verified:
      continue
    key = (destination.channel, destination.destination)
    if key in seen:
      continue
    seen.add(key)
    destinations.append(destination)
  return destinations


def build_decision_url(request: ApprovalRequest, *, base_url: str | None = None) -> str | None:
  base = (base_url or os.environ.get("GATEWAY_APPROVAL_DECISION_BASE_URL") or "").strip()
  if not base:
    return None
  run_id = _safe_segment(request.run_id or request.session_id or request.request_id)
  approval_id = _safe_segment(request.approval_id)
  return f"{base.rstrip('/')}/#agent-control/run/{run_id}/approval/{approval_id}"


def render_approval_notification_message(request: ApprovalRequest, *, base_url: str | None = None) -> str:
  product = _safe_text(os.environ.get("GATEWAY_APPROVAL_NOTIFICATION_PRODUCT_LABEL") or "Agent Control", max_length=64)
  tool_label = _safe_text(request.tool_name, max_length=96)
  approval_class = _safe_text(request.tool_class, max_length=48)
  parts = [
    f"{product} approval needed: {approval_class} action is waiting.",
    f"Tool: {tool_label}.",
  ]
  if request.expires_at is not None:
    parts.append(f"Expires at {request.expires_at.isoformat()}.")
  decision_url = build_decision_url(request, base_url=base_url)
  if decision_url:
    parts.append(f"Open: {decision_url}")
  return " ".join(parts)


def build_env_approval_notification_destination_resolver(
  env: dict[str, str] | None = None,
) -> ApprovalNotificationDestinationResolver | None:
  source = env if env is not None else os.environ
  raw = (source.get("GATEWAY_APPROVAL_NOTIFICATION_DESTINATIONS_JSON") or "").strip()
  if not raw:
    return None
  mapping = json.loads(raw)

  def _resolver(request: ApprovalRequest) -> list[ApprovalNotificationDestination]:
    user_map = mapping.get("users", mapping) if isinstance(mapping, dict) else {}
    entry = user_map.get(str(request.user_id)) if isinstance(user_map, dict) else None
    if isinstance(entry, list):
      return normalize_approval_notification_destinations(entry)
    if isinstance(entry, dict):
      return normalize_approval_notification_destinations(_destinations_from_mapping(entry))
    return []

  return _resolver


def build_env_telegram_approval_notification_sender(env: dict[str, str] | None = None) -> ApprovalNotificationSender | None:
  source = env if env is not None else os.environ
  token = (
    source.get("GATEWAY_APPROVAL_TELEGRAM_BOT_TOKEN")
    or source.get("APPROVAL_TELEGRAM_BOT_TOKEN")
    or ""
  ).strip()
  if not token:
    return None

  async def _sender(row: dict[str, Any]) -> None:
    if row.get("channel") != "telegram":
      raise RuntimeError("approval notification channel has no configured sender")
    destination = str(row.get("destination") or "").strip()
    message = str(row.get("message") or "").strip()
    if not destination or not message:
      raise RuntimeError("approval notification row is missing destination or message")
    data = urllib.parse.urlencode(
      {
        "chat_id": destination,
        "text": message,
        "disable_web_page_preview": "true",
      }
    ).encode("utf-8")
    url = f"https://api.telegram.org/bot{token}/sendMessage"

    def _post() -> None:
      request = urllib.request.Request(url, data=data, method="POST")
      with urllib.request.urlopen(request, timeout=10) as response:
        response.read()

    await asyncio.to_thread(_post)

  return _sender


async def maybe_await(value: Any) -> Any:
  if inspect.isawaitable(value):
    return await value
  return value


def _destinations_from_mapping(entry: dict[str, Any]) -> list[dict[str, Any]]:
  destinations: list[dict[str, Any]] = []
  for channel in ("telegram", "email", "push"):
    value = entry.get(channel)
    if isinstance(value, str) and value.strip():
      destinations.append({"channel": channel, "destination": value.strip(), "verified": True})
    elif isinstance(value, dict):
      destination = dict(value)
      destination.setdefault("channel", channel)
      destinations.append(destination)
  return destinations


def _safe_segment(value: str | None) -> str:
  text = str(value or "").strip()
  return urllib.parse.quote(text, safe="")


def _safe_text(value: Any, *, max_length: int) -> str:
  text = re.sub(r"\s+", " ", str(value or "")).strip()
  if len(text) <= max_length:
    return text
  return f"{text[: max(0, max_length - 1)]}..."


__all__ = [
  "ApprovalNotificationChannel",
  "ApprovalNotificationDestination",
  "ApprovalNotificationDestinationResolver",
  "ApprovalNotificationSender",
  "ApprovalNotificationState",
  "PUBLIC_APPROVAL_NOTIFICATION_CHANNELS",
  "approval_notification_policy_for_request",
  "build_decision_url",
  "build_env_approval_notification_destination_resolver",
  "build_env_telegram_approval_notification_sender",
  "maybe_await",
  "normalize_approval_notification_destinations",
  "render_approval_notification_message",
]
