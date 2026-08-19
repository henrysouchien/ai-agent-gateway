"""Canonical commercial usage payload helpers used by the runtime gateway."""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal
import hashlib
from importlib.resources import files
import json
import math
from pathlib import Path
from uuid import UUID


def packaged_usage_v3_contract_directory() -> Path:
  return Path(str(files("agent_gateway") / "contracts" / "commercial-usage-v3"))


def _normalized_usage_body(event: dict) -> dict:
  body = dict(event)
  body.pop("source_payload_sha256", None)
  for field in (
    "separately_billed_tool_cost_usd",
    "producer_estimated_cost_usd",
    "provider_reported_cost_usd",
  ):
    if body.get(field) is not None:
      body[field] = Decimal(str(body[field]))
  return body


def canonical_usage_payload_sha256(event: dict) -> str:
  body = _normalized_usage_body(event)
  return "sha256:" + hashlib.sha256(_canonical_json(body).encode("utf-8")).hexdigest()


def _canonical_json(value) -> str:
  if value is None:
    return "null"
  if value is True:
    return "true"
  if value is False:
    return "false"
  if type(value) is int:
    return str(value)
  if type(value) is float:
    if not math.isfinite(value):
      raise ValueError("commercial canonical float must be finite")
    return _canonical_json(Decimal(str(value)))
  if isinstance(value, Decimal):
    if not value.is_finite():
      raise ValueError("commercial canonical decimal must be finite")
    rendered = format(value, "f")
    if "." in rendered:
      rendered = rendered.rstrip("0").rstrip(".")
    return "0" if rendered in {"-0", ""} else rendered
  if isinstance(value, str):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
  if isinstance(value, list):
    return "[" + ",".join(_canonical_json(item) for item in value) + "]"
  if isinstance(value, dict):
    return "{" + ",".join(
      _canonical_json(key) + ":" + _canonical_json(value[key])
      for key in sorted(value)
    ) + "}"
  if isinstance(value, datetime):
    if value.tzinfo is None or value.utcoffset() is None:
      raise ValueError("commercial canonical datetime must be timezone-aware")
    rendered = value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
    return _canonical_json(rendered)
  if isinstance(value, UUID):
    return _canonical_json(str(value))
  raise ValueError(f"unsupported commercial canonical type: {type(value).__name__}")
