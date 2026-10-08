"""Startup failures the launch boundary classifies by type.

Defined apart from ``gateway_server`` on purpose: ``gateway.service`` runs that
module as ``__main__`` (``python3 -m agent_gateway.gateway_server``), while the
app imports it by name, so a class defined there exists twice and an
``isinstance`` check across the two never matches. This module is imported by
name from both sides.
"""

from __future__ import annotations

SCHEMA_INCOMPATIBLE_EXIT_CODE = 4


class GatewaySchemaIncompatibleError(RuntimeError):
  """A store the app serves is at a schema this code cannot serve.

  Raised from the app's startup; the launch boundary reports it as
  ``failure_class=schema`` with exit status 4 instead of a restartable failure.
  """
