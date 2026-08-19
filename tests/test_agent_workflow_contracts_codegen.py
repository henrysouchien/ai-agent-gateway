from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

from jsonschema import Draft202012Validator

from agent_gateway.capability_binding import CAPABILITY_RESOLUTION_CODES
from agent_workflow_contracts import parse_delivery_envelope


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


def test_generated_capability_resolution_codes_match_gateway_literal() -> None:
    """The checked-in TS constant must equal the gateway's canonical code set.

    The clients (taskpane chatContract.ts via this generated artifact,
    Discord/Telegram via the Python constant) must never hand copy the typed
    selection-refusal code list; this fails on any drift between the
    generated artifact and CapabilityResolutionCode.
    """
    source = (GENERATED / "capability-resolution-codes.ts").read_text(encoding="utf-8")
    generated_codes = re.findall(r'^\s+"([^"]+)",$', source, flags=re.MULTILINE)
    assert generated_codes == sorted(CAPABILITY_RESOLUTION_CODES)


def test_delivery_envelope_goldens_are_valid_in_python_schema_and_typescript() -> None:
    schema = json.loads(
        (GENERATED / "delivery-envelope.schema.json").read_text(encoding="utf-8")
    )
    for basename in (
        "delivery-envelope-v1-historical.golden",
        "delivery-envelope.golden",
        "delivery-envelope-truncated.golden",
    ):
        golden = json.loads((GENERATED / f"{basename}.json").read_text(encoding="utf-8"))
        assert parse_delivery_envelope(golden).model_dump(mode="json") == golden
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
                str(GENERATED / f"{basename}.ts"),
            ],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        assert result.returncode == 0, result.stdout + result.stderr


def test_historical_v1_delivery_golden_bytes_are_frozen() -> None:
    raw = (GENERATED / "delivery-envelope-v1-historical.golden.json").read_bytes()
    assert len(raw) == 2101
    assert hashlib.sha256(raw).hexdigest() == (
        "ddabbc61a8ca47dc676f7e601cc96e0dbe979c28a8209f65516f8abd57929fe9"
    )
