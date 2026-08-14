from __future__ import annotations

import importlib
from pathlib import Path
import sys

import pytest
from pydantic import ValidationError

from schema.annual_manifest_slot_registry import ANNUAL_MANIFEST_INPUT_SLOTS
from schema.build_manifest import (
    AnnualManifestInputClosure,
    BuildManifest,
    annual_manifest_input_closure_content_sha256,
)


TESTS_DIR = Path(__file__).resolve().parent
if str(TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(TESTS_DIR))
_WORLD = importlib.import_module("bma603_annual_manifest_world")


def test_bma603_annual_contract_fixture_is_strict_and_predecessor_free() -> None:
    world = _WORLD.build_bma603_annual_manifest_contract_world()

    assert type(world.manifest) is BuildManifest
    assert type(world.closure) is AnnualManifestInputClosure
    assert world.closure.model_state_ref == world.manifest.model_state_ref
    assert tuple(entry.input_slot for entry in world.closure.entries) == (
        ANNUAL_MANIFEST_INPUT_SLOTS
    )
    assert world.closure.content_sha256 == (
        annual_manifest_input_closure_content_sha256(world.closure)
    )


def test_bma603_annual_closure_rejects_slot_order_drift() -> None:
    world = _WORLD.build_bma603_annual_manifest_contract_world()
    payload = world.closure.model_dump(mode="python")
    payload["entries"] = tuple(reversed(payload["entries"]))

    with pytest.raises(
        ValidationError,
        match="exact ordered 21 slots",
    ):
        AnnualManifestInputClosure.model_validate(payload, strict=True)
