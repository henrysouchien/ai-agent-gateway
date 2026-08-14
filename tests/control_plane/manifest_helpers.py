from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from agent_gateway.autonomous_admission_ledger import prepare_autonomous_admission_ledger
from agent_gateway.autonomous_launch_envelope import AutonomousControlAuthority
from agent_gateway.capability_binding import CapabilityBind
from agent_gateway.model_registry import (
  INITIAL_MODEL_REGISTRY,
  INITIAL_MODEL_SELECTION_POLICY,
)


TASK_MANIFEST_VERSION = 7
DEFAULT_CREDENTIAL_HANDLE_ID = "service:test-product:anthropic"
_DEFAULT_MODEL_ENTRY = INITIAL_MODEL_REGISTRY.require("anthropic.claude-opus-5")


def write_v6_manifest(
  log_dir: Path,
  task_id: str,
  **overrides: Any,
) -> dict[str, Any]:
  """Write a structurally valid v6 autonomous manifest for gateway tests."""
  log_dir.mkdir(parents=True, exist_ok=True)
  operator_path = (log_dir / f"{task_id}.operator-messages.jsonl").resolve()
  operator_path.touch(exist_ok=True)
  operator_path.chmod(0o600)
  operator_stat = operator_path.stat()
  owner_lease_path = (log_dir / f"{task_id}.owner-lease").resolve()
  owner_lease_path.touch(exist_ok=True)
  owner_lease_path.chmod(0o600)
  owner_lease_stat = owner_lease_path.stat()
  ledger = prepare_autonomous_admission_ledger(
    (log_dir / ".autonomous-admission-ledger.sqlite3").resolve()
  )
  control_authority = AutonomousControlAuthority(
    control_mode="file",
    admission_ledger_path=ledger.path,
    admission_ledger_device=ledger.device,
    admission_ledger_inode=ledger.inode,
    operator_inbox_path=str(operator_path),
    operator_inbox_device=operator_stat.st_dev,
    operator_inbox_inode=operator_stat.st_ino,
    approval_decisions_path=None,
    approval_decisions_device=None,
    approval_decisions_inode=None,
    approval_store_path=None,
    approval_store_device=None,
    approval_store_inode=None,
  )
  default_bind = CapabilityBind(
    schema_version="1.0",
    capability_id="session.driver",
    model_key=_DEFAULT_MODEL_ENTRY.key,
    provider=_DEFAULT_MODEL_ENTRY.provider,
    upstream_model=_DEFAULT_MODEL_ENTRY.upstream_model,
    adapter=_DEFAULT_MODEL_ENTRY.adapter,
    protocol_profile=_DEFAULT_MODEL_ENTRY.protocol_profile,
    route=_DEFAULT_MODEL_ENTRY.route,
    effort="high",
    credential_principal="service",
    credential_ref=DEFAULT_CREDENTIAL_HANDLE_ID,
    run_mode="autonomous",
    registry_revision=INITIAL_MODEL_REGISTRY.revision,
    policy_revision=INITIAL_MODEL_SELECTION_POLICY.revision,
    selection_source="capability_default",
  )
  manifest: dict[str, Any] = {
    "manifest_version": TASK_MANIFEST_VERSION,
    "task_id": task_id,
    "control_run_id": task_id,
    "session_id": task_id,
    "channel_id": hashlib.sha256(task_id.encode("utf-8")).hexdigest(),
    "user_id": "alice",
    "user_email": "alice@example.com",
    "role": "owner",
    "profile": "analyst",
    "mode": "skill",
    "task": None,
    "skill": "earnings-review",
    "pack": None,
    "deliver": True,
    "context": "Original work packet",
    "ticker": "MSFT",
    "channel": "tui",
    "dev_mode": False,
    "cmd": ["python3", "-m", "agent.autonomous", "--profile", "analyst"],
    "log_path": str((log_dir / f"{task_id}.log").resolve()),
    "operator_inbox_path": str(operator_path),
    "approval_decisions_path": None,
    "owner_lease_path": str(owner_lease_path),
    "owner_lease_device": owner_lease_stat.st_dev,
    "owner_lease_inode": owner_lease_stat.st_ino,
    "control_authority": control_authority.receipt(),
    "started_at": 100.0,
    "state": "completed",
    "exit_code": 0,
    "error": None,
    "terminal_reason": None,
    "completed_at": 125.0,
    "resumed_from": None,
    "resumed_as": [],
    "capability_bind": default_bind.receipt(),
  }
  manifest.update(overrides)
  (log_dir / f"{task_id}.task.json").write_text(
    json.dumps(manifest, sort_keys=True) + "\n",
    encoding="utf-8",
  )
  return manifest
