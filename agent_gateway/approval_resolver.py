from __future__ import annotations

import importlib
import os
from typing import Any

from .approval_policy import ApprovalPolicy
from .single_user_policy import SingleUserApprovalPolicy


def resolve_policy(*, store: Any | None = None) -> ApprovalPolicy:
  class_path = os.getenv("GATEWAY_APPROVAL_POLICY_CLASS", "").strip()
  if not class_path:
    return SingleUserApprovalPolicy(store=store)
  module_name, sep, attr = class_path.rpartition(".")
  if not sep or not module_name or not attr:
    raise RuntimeError("GATEWAY_APPROVAL_POLICY_CLASS must be a dotted class path")
  module = importlib.import_module(module_name)
  cls = getattr(module, attr)
  try:
    return cls(store=store)
  except TypeError:
    return cls()
