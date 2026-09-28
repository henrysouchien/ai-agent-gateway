"""Where the gateway is: one environment name, one reader.

The origin of the gateway a process talks to is one fact, so it has one
environment name -- ``GATEWAY_BASE_URL`` -- and this module is the only place
that reads it or spells its default. A caller that just needs the address calls
:func:`gateway_base_url`. A caller that owns a richer precedence ladder (the
CLI's ``--base-url`` > environment > saved config > default) takes the
environment rung from :func:`configured_gateway_base_url` and the last rung from
:data:`DEFAULT_GATEWAY_BASE_URL`, so the ladder stays the caller's and the name
and the default stay here.

A second name for this fact sends a run to a gateway nobody chose, silently:
`docs/closed/gateway-address-one-name-one-resolver.md`.
"""

from __future__ import annotations

import os
from typing import Mapping


GATEWAY_BASE_URL_ENV = "GATEWAY_BASE_URL"
DEFAULT_GATEWAY_BASE_URL = "https://localhost:8000"


def configured_gateway_base_url(
  environ: Mapping[str, str] | None = None,
) -> str | None:
  """Return the configured gateway origin, or None when none is configured."""

  source = os.environ if environ is None else environ
  raw = (source.get(GATEWAY_BASE_URL_ENV) or "").strip()
  if not raw:
    return None
  # A configured-but-malformed origin -- "/" or "///" -- must stay visible to
  # the caller's own base-URL validator. Normalising it to None would make it
  # indistinguishable from unset, and the caller would fall through to
  # DEFAULT_GATEWAY_BASE_URL and drive the shared pair with no error, which is
  # the failure this module exists to remove.
  return raw.rstrip("/") or raw


def gateway_base_url(environ: Mapping[str, str] | None = None) -> str:
  """Return the origin of the gateway this process talks to."""

  return configured_gateway_base_url(environ) or DEFAULT_GATEWAY_BASE_URL
