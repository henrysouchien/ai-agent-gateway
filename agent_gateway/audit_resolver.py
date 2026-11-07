from __future__ import annotations

import importlib
import os
from pathlib import Path

from .audit_writer import AuditWriter, JSONLAuditWriter, default_audit_root


def resolve_audit_root() -> Path:
  """The JSONL approval-audit directory: GATEWAY_APPROVAL_AUDIT_DIR, else the writer default.

  The writer and anything that reads its files back (the telemetry shipper) take
  the root from here.
  """
  root = os.getenv("GATEWAY_APPROVAL_AUDIT_DIR", "").strip()
  return Path(os.path.abspath(root)) if root else default_audit_root()


def resolve_audit_writer() -> AuditWriter:
  class_path = os.getenv("GATEWAY_APPROVAL_AUDIT_CLASS", "").strip()
  if not class_path:
    return JSONLAuditWriter(resolve_audit_root())
  module_name, sep, attr = class_path.rpartition(".")
  if not sep or not module_name or not attr:
    raise RuntimeError("GATEWAY_APPROVAL_AUDIT_CLASS must be a dotted class path")
  module = importlib.import_module(module_name)
  cls = getattr(module, attr)
  return cls()
