from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

from jsonschema import Draft202012Validator

from agent_workflow_contracts import DeliveryEnvelope


ROOT = Path(__file__).resolve().parents[3]
GENERATED = ROOT / "packages" / "agent-gateway" / "agent_workflow_contracts" / "generated"


def test_checked_in_contract_clients_have_no_generation_drift() -> None:
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "generate_agent_workflow_contracts.py"), "--check"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_delivery_envelope_golden_is_valid_in_python_schema_and_typescript() -> None:
    golden = json.loads(
        (GENERATED / "delivery-envelope.golden.json").read_text(encoding="utf-8")
    )
    assert DeliveryEnvelope.model_validate(golden).model_dump(mode="json") == golden
    schema = json.loads(
        (GENERATED / "delivery-envelope.schema.json").read_text(encoding="utf-8")
    )
    Draft202012Validator(schema).validate(golden)

    result = subprocess.run(
        [
            "npx",
            "tsc",
            "--noEmit",
            "--skipLibCheck",
            "--target",
            "ES2023",
            "--module",
            "NodeNext",
            "--moduleResolution",
            "NodeNext",
            str(GENERATED / "delivery-envelope.golden.ts"),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
