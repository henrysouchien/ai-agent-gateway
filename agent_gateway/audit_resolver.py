from __future__ import annotations

import importlib
import os

from .audit_writer import AuditWriter, JSONLAuditWriter


def resolve_audit_writer() -> AuditWriter:
  class_path = os.getenv("GATEWAY_APPROVAL_AUDIT_CLASS", "").strip()
  if not class_path:
    return JSONLAuditWriter()
  module_name, sep, attr = class_path.rpartition(".")
  if not sep or not module_name or not attr:
    raise RuntimeError("GATEWAY_APPROVAL_AUDIT_CLASS must be a dotted class path")
  module = importlib.import_module(module_name)
  cls = getattr(module, attr)
  return cls()
