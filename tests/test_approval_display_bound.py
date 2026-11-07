"""An exact-write approval's display copy stays inside the approval audit record.

The approval request persists its display arguments, and the audit writer
refuses a record past its byte limit. A plan that carries a whole filing must
still be approvable: the display copy abbreviates long strings with their
length and SHA-256, while the approval hash keeps binding the exact arguments.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from pathlib import Path

import pytest

from agent_gateway.approval_audit import build_audit_entry
from agent_gateway.approval_policy import (
  APPROVAL_DISPLAY_MAX_STRING_CHARS,
  RunContext,
  bounded_approval_display,
  build_approval_request,
  sha256_args,
)
from agent_gateway.audit_writer import JSONLAuditWriter
from agent_gateway.mcp_client import RegisteredMcpRawPatchAuthorization
from agent_gateway.registry_scope import RegistryScope


# The text a whole-filing read registers, sized past the audit record limit.
_WHOLE_FILING = "Item 1. Business. " * 30_000


def _plan_arguments() -> dict[str, object]:
  return {
    "research_file_id": 29,
    "requested_ops": [
      {"op": "register_sources", "source_ref": "S1", "quote": "223,000 people"},
    ],
    "canonical_ops": {
      "sources": [
        {
          "id": "src_1",
          "excerpts": [{"handle_id": "h_994fbb770cdd", "text": _WHOLE_FILING}],
        }
      ],
    },
  }


def _authorization(arguments: dict[str, object]) -> RegisteredMcpRawPatchAuthorization:
  return RegisteredMcpRawPatchAuthorization(
    approval_identity={"identity_source": "reviewed_change_binding"},
    approval_arguments=arguments,
    approval_arguments_hash=sha256_args(arguments),
    prepared_payload=b'{"prepared":true}',
  )


def _audit_entry(arguments: dict[str, object], *, args_hash: str):
  request = build_approval_request(
    tool_call_id="call-1",
    tool_name="apply_patch_ops",
    tool_class="state_write",
    tool_args_redacted=arguments,
    args_hash=args_hash,
    run_context=RunContext(
      user_id="alice",
      request_id="request-1",
      registry_scope=RegistryScope("run", "test-run", "/tmp/test-store"),
    ),
  )
  return build_audit_entry(
    raw_tool_args={},
    deployment_secret=b"secret",
    key_id="key-1",
    event_type="request_created",
    request=request,
  )


def test_whole_filing_plan_display_fits_the_audit_record(tmp_path: Path) -> None:
  arguments = _plan_arguments()
  authorization = _authorization(arguments)

  entry = _audit_entry(
    authorization.materialize_approval_display(),
    args_hash=authorization.approval_arguments_hash,
  )
  writer = JSONLAuditWriter(tmp_path)
  asyncio.run(writer.write(entry))

  persisted, _ = asyncio.run(writer.query(owner_user_id="alice", limit=10))
  assert [row.entry_id for row in persisted] == [entry.entry_id]
  assert persisted[0].args_hash == sha256_args(arguments)


def test_the_exact_arguments_alone_exceed_the_audit_record(tmp_path: Path) -> None:
  # The control: the failure the bounded display exists to prevent.
  arguments = _plan_arguments()
  entry = _audit_entry(arguments, args_hash=sha256_args(arguments))

  with pytest.raises(RuntimeError, match="record exceeds its byte limit"):
    asyncio.run(JSONLAuditWriter(tmp_path).write(entry))


def test_display_keeps_structure_short_values_and_the_full_text_identity() -> None:
  arguments = _plan_arguments()
  authorization = _authorization(arguments)

  display = authorization.materialize_approval_display()

  assert display["research_file_id"] == 29
  assert display["requested_ops"] == arguments["requested_ops"]
  excerpt = display["canonical_ops"]["sources"][0]["excerpts"][0]  # type: ignore[index]
  assert excerpt["handle_id"] == "h_994fbb770cdd"
  head, marker = excerpt["text"].split("…", 1)
  assert head == _WHOLE_FILING[:APPROVAL_DISPLAY_MAX_STRING_CHARS]
  assert marker == (
    f"[truncated: {len(_WHOLE_FILING)} chars, "
    f"sha256:{hashlib.sha256(_WHOLE_FILING.encode()).hexdigest()}]"
  )
  # The binding stays on the exact arguments, untouched by the display.
  assert authorization.materialize_approval_arguments() == arguments
  assert authorization.approval_arguments_hash == sha256_args(arguments)
  assert len(json.dumps(display)) < 16 * 1024


def test_a_string_at_the_bound_is_not_abbreviated() -> None:
  at_bound = "x" * APPROVAL_DISPLAY_MAX_STRING_CHARS

  assert bounded_approval_display({"k": [at_bound, 7, None, True]}) == {
    "k": [at_bound, 7, None, True]
  }
