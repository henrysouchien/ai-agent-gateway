"""Production send authority for outbound notifications.

A process may reach a production notification destination (the operator's or a
user's Telegram chat) only when it runs from one of the two production
installs, decided by where this module's own file sits:

- the shared local gateway runtime,
  ``<login home>/.local/state/hank/local-gateway-runtime/versions/<version>/ai-excel-addin/``,
  which ``local_gateway_bootstrap`` execs every launchd lane and the shared
  gateway pair from; and
- the hosted EC2 deploy roots: ``/var/www/agent_gateway``, where
  ``scripts/deploy_ec2_gateway.sh`` installs the tree and a venv holding this
  package as a wheel and ``deploy/systemd`` runs every unit, and
  ``/var/www/investment_tools``, the hosted Investment runtime's own venv.

Isolated instances run from their own ``isolated-gateways/<N>/runtime/versions/...``
tree, a checkout runs from the checkout, and the EC2 test box runs from the
user's checkout under ``/home``; all are held.

The evidence is this file's location (plus the login account's home from the
account database for the local runtime). It reads no environment variable, so
neither a dotenv (``bootstrap_env`` loads two) nor SSM hydration nor the caller's
shell can supply it.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

HOLD_REASON = "dev_run"

_CODE_FILE = Path(__file__).resolve()
# The hosted install roots. The gateway root's owners are scripts/deploy_ec2_gateway.sh
# (`current=`) and every unit under deploy/systemd (WorkingDirectory and ExecStart). The
# investment root is the hosted Investment runtime: its own venv holds agent_gateway as a
# wheel, and deploy/mcp.production.json runs jobs-mcp (whose notifications are a sender)
# and deploy/investment-runtime/investment-quant.service runs the capability worker from
# it. tests/notify/test_production_send_hold.py pins both to these values. Revert site:
# remove an entry here (the EC2 branch of is_production_runtime() follows the tuple).
_EC2_DEPLOY_ROOTS = (Path("/var/www/agent_gateway"), Path("/var/www/investment_tools"))
_LOCAL_PACKAGE_DIR = "ai-excel-addin"


class ProductionSendHeld(RuntimeError):
  """The sender is not the production runtime; nothing was sent."""


def _shared_runtime_root() -> Path | None:
  """The local shared runtime root, or None when this install carries no repo tree.

  The path owner is ``scripts/local_machine_paths.py`` in the same ai-excel-addin
  tree as this package. An install without that tree (the EC2 wheel) is not a
  local runtime version, so there is no root to compare against.
  """

  parents = Path(__file__).resolve().parents
  if len(parents) < 4:
    return None
  path = parents[3] / "scripts" / "local_machine_paths.py"
  if not path.is_file():
    return None
  spec = importlib.util.spec_from_file_location("production_identity_machine_paths", path)
  if spec is None or spec.loader is None:
    raise OSError(f"cannot load the machine-paths owner: {path}")
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return Path(module.gateway_runtime_root()).resolve(strict=False)


def _under(path: Path, root: Path) -> bool:
  try:
    path.relative_to(root)
  except ValueError:
    return False
  return True


def is_production_runtime() -> bool:
  """True only when this code runs from a production install."""

  if any(_under(_CODE_FILE, root) for root in _EC2_DEPLOY_ROOTS):
    return True
  root = _shared_runtime_root()
  if root is None:
    return False
  try:
    relative = _CODE_FILE.relative_to(root / "versions")
  except ValueError:
    return False
  return len(relative.parts) > 2 and relative.parts[1] == _LOCAL_PACKAGE_DIR


def require_production_runtime(channel: str) -> None:
  """Raise ``ProductionSendHeld`` unless this process is a production runtime."""

  if not is_production_runtime():
    raise ProductionSendHeld(f"{channel} send held: {HOLD_REASON}")


__all__ = [
  "HOLD_REASON",
  "ProductionSendHeld",
  "is_production_runtime",
  "require_production_runtime",
]
