from __future__ import annotations

import importlib
import shutil
from pathlib import Path

import yaml

from model_authority import current
from model_authority.loader import PACKAGED_AUTHORITY_DIR, load_model_authority
from agent_gateway import easy
from gateway_test_support import model_defaults


def test_tier_driver_projects_packaged_identity(tmp_path: Path, monkeypatch) -> None:
  expected = model_defaults.capability_default("session.driver")
  document = yaml.safe_load((PACKAGED_AUTHORITY_DIR / "model-authority.yaml").read_text())
  document["schema"] = "model-authority/2.2"
  document["selection"]["tiers"] = {
    "deep": {"default": {"models": [expected.model_key], "effort": expected.effort}, "allowed_model_keys": [expected.model_key]},
    "default": {"default": {"alias": "deep"}},
  }
  document["selection"]["capabilities"]["session.driver"]["default"] = {
    "kind": "tier", "tier": "default",
  }
  directory = tmp_path / "authority"
  shutil.copytree(PACKAGED_AUTHORITY_DIR, directory)
  (directory / "model-authority.yaml").write_text(yaml.safe_dump(document, sort_keys=False))
  authority = load_model_authority(directory)
  with monkeypatch.context() as patch:
    patch.setattr(current, "INITIAL_MODEL_REGISTRY", authority.registry)
    patch.setattr(current, "INITIAL_MODEL_SELECTION_POLICY", authority.selection_policy)
    importlib.reload(easy)
    patch.setattr(model_defaults, "INITIAL_MODEL_REGISTRY", authority.registry)
    patch.setattr(model_defaults, "INITIAL_MODEL_SELECTION_POLICY", authority.selection_policy)
    actual = model_defaults.capability_default("session.driver")
    assert easy._DEFAULT_EASY_MODEL_KEY == expected.model_key
    assert (actual.model_key, actual.effort) == (expected.model_key, expected.effort)
  importlib.reload(easy)
