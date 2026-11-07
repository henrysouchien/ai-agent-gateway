"""The registry scope a tool call is declared under, before the tool runs.

One value per call, set where its ``RunContext`` is constructed and carried by
value through ``ToolExecutionContext`` and ``ToolResultContext`` into FMS
``RunMetadata`` (docs/design/registry-scope-carrier-design.md). It is never
inferred after the tool runs and never unioned with another scope.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class RegistryScope:
  """A persisted source registry: ``session`` rows or one run's rows."""

  scope_type: Literal["session", "run"]
  # The session id for "session"; the run id the rows are written under for "run".
  scope_id: str
  # The registry root the rows live in.
  store_dir: str


@dataclass(frozen=True)
class NoRegistryScope:
  """Declared by a run whose constructor holds no source store: rows stay in memory.

  A declaration of the evidence the constructor holds, not a default. A run
  without a storage owner declares it where its ``RunContext`` or FMS
  ``RunMetadata`` is built (interactive turns without a workspace, the
  embedding SDK runner, workspace-less autonomous and patch routes), and a
  pre-carrier approval binding that is replayed rather than executed. Every
  registry reader handles it: ``S_N`` resolves ``registry_missing`` and
  nothing persists.
  """


DeclaredRegistryScope = RegistryScope | NoRegistryScope


__all__ = ["DeclaredRegistryScope", "NoRegistryScope", "RegistryScope"]
