from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

import pytest
from starlette.requests import Request

from agent_gateway.agent_session_log import AgentSessionLog
from agent_gateway.server_workflow_output_routes import workflow_output_response
from agent_gateway.session import GatewaySession
from agent_gateway.workflow_output_attachment import WorkflowOutputAttachment
from agent_workflow_contracts import (
  AuthoredDeliverySummary,
  ContentHandle,
  ContractRef,
  DeliveryEnvelope,
  DeliveryPrimary,
  PublishedOutputRef,
)


class _Auth:
  def __init__(self, session: GatewaySession) -> None:
    self.session = session

  def verify_token(self, token: str) -> GatewaySession:
    assert token == "session-token"
    return self.session


def _request() -> Request:
  return Request({
    "type": "http",
    "method": "GET",
    "path": "/api/workflow-outputs/workflow-1/output",
    "headers": [(b"authorization", b"Bearer session-token")],
  })


def _session(tmp_path: Path) -> tuple[GatewaySession, AgentSessionLog]:
  session = GatewaySession(
    session_id="session-1",
    api_key_hash="hash",
    created_at=1,
    expires_at=4_000_000_000,
    user_id="alice",
  )
  session_log = AgentSessionLog(
    tmp_path / "session.jsonl",
    gateway_session=session,
  )
  session.agent_session_log = session_log  # type: ignore[attr-defined]
  return session, session_log


def _attachment(content: str) -> WorkflowOutputAttachment:
  payload = content.encode("utf-8")
  digest = hashlib.sha256(payload).hexdigest()
  primary_contract = ContractRef(
    namespace="workflow-output",
    name="synthesis",
    version="1.0",
    digest=f"sha256:{'a' * 64}",
  )
  primary = PublishedOutputRef(
    output_id="wout:workflow-1:phase:2:revision:1:synthesis",
    contract=primary_contract,
    content=ContentHandle(
      content_id=f"sha256:{digest}",
      content_sha256=digest,
      content_chars=len(content),
      content_bytes=len(payload),
      contract=primary_contract,
      media_type="application/json",
      encoding="canonical-json",
      retention="durable",
    ),
  )
  summary_text = "Executive summary"
  summary_payload = summary_text.encode("utf-8")
  summary_digest = hashlib.sha256(summary_payload).hexdigest()
  summary_contract = ContractRef(
    namespace="workflow-output",
    name="delivery-summary",
    version="1.0",
    digest=f"sha256:{'c' * 64}",
  )
  return WorkflowOutputAttachment(
    envelope=DeliveryEnvelope(
      workflow_run_id="workflow-1",
      phase_number=2,
      revision=1,
      summary=AuthoredDeliverySummary(
        text=summary_text,
        source=PublishedOutputRef(
          output_id=(
            "wout:workflow-1:phase:2:revision:1:delivery_summary"
          ),
          contract=summary_contract,
          content=ContentHandle(
            content_id=f"sha256:{summary_digest}",
            content_sha256=summary_digest,
            content_chars=len(summary_text),
            content_bytes=len(summary_payload),
            contract=summary_contract,
            media_type="text/plain; charset=utf-8",
            encoding="utf-8",
            retention="durable",
          ),
        ),
      ),
      primary=DeliveryPrimary(
        name="synthesis",
        published_output_ref=primary,
      ),
    ),
  )


def _record_attachment(
  session_log: AgentSessionLog,
  attachment: WorkflowOutputAttachment,
) -> None:
  session_log.append_sync({
    "type": "assistant_message",
    "content_blocks": [{"type": "text", "text": "Executive summary"}],
    "workflow_output_attachments": [attachment.to_dict()],
  })


def _reader(content: str, attachment: WorkflowOutputAttachment):
  async def read(
    workflow_run_id: str,
    output_id: str,
    after_char: int,
  ) -> dict[str, object]:
    assert workflow_run_id == attachment.workflow_run_id
    assert output_id == attachment.output_id
    chunk = content[after_char : after_char + 7]
    next_after = after_char + len(chunk)
    end = next_after == len(content)
    return {
      "ok": True,
      "action": "output",
      "workflow_run_id": workflow_run_id,
      "output_id": output_id,
      "view": "paged_exact_content",
      "source": attachment.published_output_ref.content.model_dump(mode="json"),
      "authorization": {
        "kind": "published_output",
        "workflow_run_id": workflow_run_id,
        "phase_number": attachment.delivery_phase_number,
        "revision": attachment.delivery_revision,
        "output_id": output_id,
      },
      "encoding": "canonical-json",
      "after_char": after_char,
      "content": chunk,
      "content_sha256": attachment.content_sha256,
      "content_chars": attachment.content_chars,
      "content_bytes": attachment.content_bytes,
      "next_cursor": None if end else {"after_char": next_after},
      "end": end,
    }

  return read


@pytest.mark.asyncio
async def test_authenticated_route_materializes_and_verifies_exact_output(
  tmp_path: Path,
) -> None:
  canonical = json.dumps(
    "Exact narrative with unicode π",
    ensure_ascii=False,
    separators=(",", ":"),
  )
  session, session_log = _session(tmp_path)
  attachment = _attachment(canonical)
  _record_attachment(session_log, attachment)
  session.workflow_output_reader = _reader(canonical, attachment)

  response = await workflow_output_response(
    _request(),
    attachment.workflow_run_id,
    attachment.output_id,
    auth=_Auth(session),  # type: ignore[arg-type]
    download=True,
  )

  assert response.status_code == 200
  assert response.body == canonical.encode("utf-8")
  assert response.headers["x-content-sha256"] == attachment.content_sha256
  assert response.headers["x-workflow-output-id"] == attachment.output_id
  assert response.headers["content-disposition"] == (
    'attachment; filename="synthesis.json"'
  )


@pytest.mark.asyncio
async def test_route_does_not_read_output_absent_from_authenticated_session(
  tmp_path: Path,
) -> None:
  canonical = '"secret"'
  session, _session_log = _session(tmp_path)
  attachment = _attachment(canonical)
  called = False

  async def read(*_args: object) -> object:
    nonlocal called
    called = True
    raise AssertionError("reader must not be called")

  session.workflow_output_reader = read  # type: ignore[assignment]
  response = await workflow_output_response(
    _request(),
    attachment.workflow_run_id,
    attachment.output_id,
    auth=_Auth(session),  # type: ignore[arg-type]
  )

  assert response.status_code == 404
  assert called is False


@pytest.mark.asyncio
async def test_bearer_session_cannot_read_another_sessions_attachment(
  tmp_path: Path,
) -> None:
  canonical = '"owned by alice"'
  owner, owner_log = _session(tmp_path / "owner")
  attachment = _attachment(canonical)
  _record_attachment(owner_log, attachment)
  owner.workflow_output_reader = _reader(canonical, attachment)

  other, _other_log = _session(tmp_path / "other")
  other.session_id = "session-2"
  response = await workflow_output_response(
    _request(),
    attachment.workflow_run_id,
    attachment.output_id,
    auth=_Auth(other),  # type: ignore[arg-type]
  )

  assert response.status_code == 404
  assert b"workflow_output_not_found" in response.body


@pytest.mark.asyncio
async def test_route_reports_missing_session_stable_reader(
  tmp_path: Path,
) -> None:
  canonical = '"canonical"'
  session, session_log = _session(tmp_path)
  attachment = _attachment(canonical)
  _record_attachment(session_log, attachment)

  response = await workflow_output_response(
    _request(),
    attachment.workflow_run_id,
    attachment.output_id,
    auth=_Auth(session),  # type: ignore[arg-type]
  )

  assert response.status_code == 503
  assert b"workflow_output_unavailable" in response.body


@pytest.mark.asyncio
async def test_route_rejects_materialized_bytes_that_conflict_with_attachment(
  tmp_path: Path,
) -> None:
  canonical = '"canonical"'
  session, session_log = _session(tmp_path)
  attachment = _attachment(canonical)
  _record_attachment(session_log, attachment)
  session.workflow_output_reader = _reader('"tampered!"', attachment)

  response = await workflow_output_response(
    _request(),
    attachment.workflow_run_id,
    attachment.output_id,
    auth=_Auth(session),  # type: ignore[arg-type]
  )

  assert response.status_code == 409
  assert b"workflow_output_integrity_failed" in response.body


@pytest.mark.parametrize(
  ("field_name", "field_value"),
  [
    ("content_sha256", "c" * 64),
    ("content_bytes", 999),
  ],
)
@pytest.mark.asyncio
async def test_route_rejects_page_hash_or_size_that_conflicts_with_attachment(
  tmp_path: Path,
  field_name: str,
  field_value: object,
) -> None:
  canonical = '"canonical"'
  session, session_log = _session(tmp_path)
  attachment = _attachment(canonical)
  _record_attachment(session_log, attachment)
  exact_reader = _reader(canonical, attachment)

  async def tampered_reader(
    workflow_run_id: str,
    output_id: str,
    after_char: int,
  ) -> dict[str, object]:
    page = await exact_reader(workflow_run_id, output_id, after_char)
    page[field_name] = field_value
    return page

  session.workflow_output_reader = tampered_reader
  response = await workflow_output_response(
    _request(),
    attachment.workflow_run_id,
    attachment.output_id,
    auth=_Auth(session),  # type: ignore[arg-type]
  )

  assert response.status_code == 409
  assert b"workflow_output_integrity_failed" in response.body


def _tamper_view(page: dict[str, object]) -> None:
  page["view"] = "wrong_view"


def _tamper_source(page: dict[str, object]) -> None:
  page["source"] = {"tampered": True}


def _tamper_authorization(page: dict[str, object]) -> None:
  authorization = dict(page["authorization"])  # type: ignore[arg-type]
  authorization["revision"] = 99
  page["authorization"] = authorization


def _tamper_content_identity(page: dict[str, object]) -> None:
  page["content_sha256"] = "c" * 64


def _tamper_cursor_echo(page: dict[str, object]) -> None:
  page["after_char"] = 999


def _tamper_empty_page(page: dict[str, object]) -> None:
  page["content"] = ""


def _tamper_terminal_cursor(page: dict[str, object]) -> None:
  if page["end"] is True:
    page["next_cursor"] = {"after_char": 999}


def _tamper_continuation_cursor(page: dict[str, object]) -> None:
  if page["end"] is False:
    page["end"] = "later"


def _tamper_paging_advance(page: dict[str, object]) -> None:
  if page["end"] is False:
    page["next_cursor"] = {"after_char": 3}


@pytest.mark.parametrize(
  ("expected_reason", "tamper"),
  [
    ("page_envelope_mismatch", _tamper_view),
    ("source_handle_mismatch", _tamper_source),
    ("authorization_mismatch", _tamper_authorization),
    ("content_identity_mismatch", _tamper_content_identity),
    ("paging_cursor_mismatch", _tamper_cursor_echo),
    ("empty_page", _tamper_empty_page),
    ("terminal_cursor_invalid", _tamper_terminal_cursor),
    ("continuation_cursor_invalid", _tamper_continuation_cursor),
    ("paging_advance_mismatch", _tamper_paging_advance),
  ],
)
@pytest.mark.asyncio
async def test_each_integrity_variant_logs_its_distinct_internal_reason(
  tmp_path: Path,
  caplog: pytest.LogCaptureFixture,
  expected_reason: str,
  tamper: object,
) -> None:
  canonical = '"canonical"'
  session, session_log = _session(tmp_path)
  attachment = _attachment(canonical)
  _record_attachment(session_log, attachment)
  exact_reader = _reader(canonical, attachment)

  async def tampered_reader(
    workflow_run_id: str,
    output_id: str,
    after_char: int,
  ) -> dict[str, object]:
    page = await exact_reader(workflow_run_id, output_id, after_char)
    tamper(page)  # type: ignore[operator]
    return page

  session.workflow_output_reader = tampered_reader
  with caplog.at_level(
    logging.WARNING,
    logger="agent_gateway.server_workflow_output_routes",
  ):
    response = await workflow_output_response(
      _request(),
      attachment.workflow_run_id,
      attachment.output_id,
      auth=_Auth(session),  # type: ignore[arg-type]
    )

  assert response.status_code == 409
  assert json.loads(response.body) == {
    "error": "workflow_output_integrity_failed",
    "message": "Canonical workflow output failed identity verification.",
  }
  records = [
    record
    for record in caplog.records
    if getattr(record, "data", {}).get("event")
    == "workflow_output_integrity_failed"
  ]
  assert [record.data["reason_code"] for record in records] == [
    expected_reason
  ]
  assert records[0].data == {
    "event": "workflow_output_integrity_failed",
    "stage": "materialize_verified_content",
    "reason_code": expected_reason,
    "workflow_run_id": attachment.workflow_run_id,
    "output_id": attachment.output_id,
    "phase_number": attachment.delivery_phase_number,
    "revision": attachment.delivery_revision,
    "output_name": "synthesis",
  }


@pytest.mark.asyncio
async def test_invalid_page_shape_logs_its_internal_reason(
  tmp_path: Path,
  caplog: pytest.LogCaptureFixture,
) -> None:
  canonical = '"canonical"'
  session, session_log = _session(tmp_path)
  attachment = _attachment(canonical)
  _record_attachment(session_log, attachment)

  async def invalid_reader(*_args: object) -> object:
    return 42

  session.workflow_output_reader = invalid_reader  # type: ignore[assignment]
  with caplog.at_level(
    logging.WARNING,
    logger="agent_gateway.server_workflow_output_routes",
  ):
    response = await workflow_output_response(
      _request(),
      attachment.workflow_run_id,
      attachment.output_id,
      auth=_Auth(session),  # type: ignore[arg-type]
    )

  assert response.status_code == 409
  assert [
    record.data["reason_code"]
    for record in caplog.records
    if getattr(record, "data", {}).get("event")
    == "workflow_output_integrity_failed"
  ] == ["invalid_page_shape"]


@pytest.mark.asyncio
async def test_materialized_hash_mismatch_logs_its_internal_reason(
  tmp_path: Path,
  caplog: pytest.LogCaptureFixture,
) -> None:
  canonical = '"canonical"'
  session, session_log = _session(tmp_path)
  attachment = _attachment(canonical)
  _record_attachment(session_log, attachment)
  session.workflow_output_reader = _reader('"tampered!"', attachment)

  with caplog.at_level(
    logging.WARNING,
    logger="agent_gateway.server_workflow_output_routes",
  ):
    response = await workflow_output_response(
      _request(),
      attachment.workflow_run_id,
      attachment.output_id,
      auth=_Auth(session),  # type: ignore[arg-type]
    )

  assert response.status_code == 409
  assert [
    record.data["reason_code"]
    for record in caplog.records
    if getattr(record, "data", {}).get("event")
    == "workflow_output_integrity_failed"
  ] == ["materialized_content_mismatch"]


@pytest.mark.asyncio
async def test_durable_attachment_conflict_logs_lookup_stage_reason(
  tmp_path: Path,
  caplog: pytest.LogCaptureFixture,
) -> None:
  canonical = '"canonical"'
  session, session_log = _session(tmp_path)
  attachment = _attachment(canonical)
  _record_attachment(session_log, attachment)
  summary = attachment.envelope.summary
  assert summary is not None
  conflicting_text = "Different executive summary"
  conflicting_payload = conflicting_text.encode("utf-8")
  conflicting_digest = hashlib.sha256(conflicting_payload).hexdigest()
  conflicting = WorkflowOutputAttachment(
    envelope=attachment.envelope.model_copy(
      update={
        "summary": AuthoredDeliverySummary(
          text=conflicting_text,
          source=PublishedOutputRef(
            output_id=summary.source.output_id,
            contract=summary.source.contract,
            content=ContentHandle(
              content_id=f"sha256:{conflicting_digest}",
              content_sha256=conflicting_digest,
              content_chars=len(conflicting_text),
              content_bytes=len(conflicting_payload),
              contract=summary.source.contract,
              media_type="text/plain; charset=utf-8",
              encoding="utf-8",
              retention="durable",
            ),
          ),
        ),
      }
    )
  )
  _record_attachment(session_log, conflicting)
  session.workflow_output_reader = _reader(canonical, attachment)

  with caplog.at_level(
    logging.WARNING,
    logger="agent_gateway.server_workflow_output_routes",
  ):
    response = await workflow_output_response(
      _request(),
      attachment.workflow_run_id,
      attachment.output_id,
      auth=_Auth(session),  # type: ignore[arg-type]
    )

  assert response.status_code == 409
  assert json.loads(response.body) == {
    "error": "workflow_output_integrity_failed",
    "message": "Canonical workflow output failed identity verification.",
  }
  records = [
    record
    for record in caplog.records
    if getattr(record, "data", {}).get("event")
    == "workflow_output_integrity_failed"
  ]
  assert [record.data["reason_code"] for record in records] == [
    "durable_attachment_identity_conflict"
  ]
  assert records[0].data["stage"] == "durable_attachment_lookup"
