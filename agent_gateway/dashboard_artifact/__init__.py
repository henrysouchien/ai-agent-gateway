from __future__ import annotations

from .registry import MODULE_REGISTRY, ModuleSpec


def __getattr__(name: str):
  if name in {"build_dashboard_artifact", "validate_dashboard_payload"}:
    from .qa import build_dashboard_artifact, validate_dashboard_payload

    return {
      "build_dashboard_artifact": build_dashboard_artifact,
      "validate_dashboard_payload": validate_dashboard_payload,
    }[name]
  raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
  "MODULE_REGISTRY",
  "ModuleSpec",
  "build_dashboard_artifact",
  "validate_dashboard_payload",
]
