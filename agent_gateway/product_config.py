"""Soft env-var accessor for PRODUCT_ID.

Library-safe: importing this module and calling ``gateway_product_id()`` never
raises when PRODUCT_ID is unset. App entrypoints call
``validate_product_id_or_raise()`` for fail-fast startup validation.
"""

from __future__ import annotations

import os
import re
from functools import lru_cache

_PRODUCT_ID_RE = re.compile(r"^[a-z][a-z0-9_-]{0,31}$")


@lru_cache(maxsize=1)
def gateway_product_id() -> str | None:
  raw = os.environ.get("PRODUCT_ID", "").strip()
  if not raw or not _PRODUCT_ID_RE.match(raw):
    return None
  return raw


def validate_product_id_or_raise() -> str:
  """App-layer fail-fast. Never call from package import/library paths."""
  raw = os.environ.get("PRODUCT_ID", "").strip()
  if not raw:
    raise RuntimeError(
      "PRODUCT_ID env var is required. Set in gateway.service + "
      "analyst-*@.service + analyst-telegram-bot.service."
    )
  if not _PRODUCT_ID_RE.match(raw):
    raise RuntimeError(
      f"PRODUCT_ID={raw!r} does not match ^[a-z][a-z0-9_-]{{0,31}}$"
    )
  gateway_product_id.cache_clear()
  return raw
