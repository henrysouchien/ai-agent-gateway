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
  DeliveryEnvelopeV1,
  DeliveryPrimary,
  OwnerBinding,
  PublishedOutput,
  PublishedOutputPageAuthorization,
  PublishedOutputRef,
  WorkflowContentCursor,
  WorkflowContentPage,
)

# Real pages come from the api-side store and published-output reader — the
# exact production seam behind the wired read_output.
try:
  from api.agent.orchestration.workflow_content import (
    AuthorizedPublishedOutputReader,
    WorkspaceWorkflowContentStore,
  )
except ModuleNotFoundError:  # pragma: no cover - depends on invocation path
  from agent.orchestration.workflow_content import (
    AuthorizedPublishedOutputReader,
    WorkspaceWorkflowContentStore,
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
      encoding="utf-8",
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
    envelope=DeliveryEnvelopeV1(
      schema_version="1.0",
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


def _published_reader(
  tmp_path: Path,
  content: str,
  attachment: WorkflowOutputAttachment,
):
  """Serve real pages through the production store and authorized reader."""

  workspace = tmp_path / "content-workspace"
  workspace.mkdir(exist_ok=True)
  store = WorkspaceWorkflowContentStore(workspace)
  owner = OwnerBinding(
    tenant_id="tenant-1",
    workflow_run_id=attachment.workflow_run_id,
  )
  handle = store.publish_serialized(
    content,
    contract=attachment.published_output_ref.contract,
    owner=owner,
    media_type=attachment.media_type,
  )
  publication = PublishedOutput(
    name=attachment.output_name,
    output_id=attachment.output_id,
    contract=attachment.published_output_ref.contract,
    content=handle,
  )
  reader = AuthorizedPublishedOutputReader(
    payload_reader=store,
    owner=owner,
    principal_id="alice",
    workflow_run_id=attachment.workflow_run_id,
    phase_number=attachment.delivery_phase_number,
    revision=attachment.delivery_revision,
    publications=(publication,),
  )

  async def read(
    workflow_run_id: str,
    output_id: str,
    after_char: int,
  ) -> WorkflowContentPage:
    assert workflow_run_id == attachment.workflow_run_id
    assert output_id == attachment.output_id
    return reader.page(
      publication,
      owner=owner,
      principal_id="alice",
      after_char=after_char,
    )

  return read


def _content_page(
  content: str,
  attachment: WorkflowOutputAttachment,
  after_char: int,
  *,
  source: ContentHandle | None = None,
  authorization: PublishedOutputPageAuthorization | None = None,
) -> WorkflowContentPage:
  """Build an internally-valid typed page claiming the given identity."""

  chunk = content[after_char : after_char + 7]
  next_after = after_char + len(chunk)
  end = next_after == len(content)
  return WorkflowContentPage(
    source=source or attachment.published_output_ref.content,
    authorization=authorization
    or PublishedOutputPageAuthorization(
      workflow_run_id=attachment.workflow_run_id,
      phase_number=attachment.delivery_phase_number,
      revision=attachment.delivery_revision,
      output_id=attachment.output_id,
    ),
    after_char=after_char,
    content=chunk,
    next_cursor=(
      None if end else WorkflowContentCursor(after_char=next_after)
    ),
    end=end,
    complete_source=after_char == 0 and end,
  )


def _lying_page_reader(content: str, attachment: WorkflowOutputAttachment):
  """Typed pages claiming the attachment's exact source over different text.

  Every per-page check passes; only the final joined-digest defense can
  catch the substitution.
  """

  async def read(
    workflow_run_id: str,
    output_id: str,
    after_char: int,
  ) -> WorkflowContentPage:
    assert workflow_run_id == attachment.workflow_run_id
    assert output_id == attachment.output_id
    return _content_page(content, attachment, after_char)

  return read


def _variant_source(
  attachment: WorkflowOutputAttachment,
  **overrides: object,
) -> ContentHandle:
  dump = attachment.published_output_ref.content.model_dump(mode="json")
  dump.update(overrides)
  if "content_sha256" in overrides:
    dump["content_id"] = f"sha256:{overrides['content_sha256']}"
  return ContentHandle.model_validate(dump)


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
  session.workflow_output_reader = _published_reader(
    tmp_path, canonical, attachment
  )

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
async def test_acc_e2e_01_typed_service_pages_materialize_across_cursors(
  tmp_path: Path,
) -> None:
  """Regression for ACC-E2E-01 (the deterministic envelope-dialect 409).

  The wired reader returns bare typed WorkflowContentPage values — never a
  workflow_run TOOL envelope — so real store-produced pages for a valid
  attachment must materialize end to end across multiple content pages.
  """

  canonical = json.dumps(
    {"narrative": "n" * 30_000},
    ensure_ascii=False,
    separators=(",", ":"),
  )
  session, session_log = _session(tmp_path)
  attachment = _attachment(canonical)
  _record_attachment(session_log, attachment)
  session.workflow_output_reader = _published_reader(
    tmp_path, canonical, attachment
  )

  response = await workflow_output_response(
    _request(),
    attachment.workflow_run_id,
    attachment.output_id,
    auth=_Auth(session),  # type: ignore[arg-type]
  )

  assert response.status_code == 200
  assert response.body == canonical.encode("utf-8")
  assert response.headers["x-content-sha256"] == attachment.content_sha256


@pytest.mark.asyncio
async def test_legacy_workflow_run_tool_envelope_dialect_is_rejected(
  tmp_path: Path,
  caplog: pytest.LogCaptureFixture,
) -> None:
  """The old workflow_run TOOL-envelope dialect is not what the wired reader
  produces; an untyped mapping speaking it must fail as an invalid page."""

  canonical = '"canonical"'
  session, session_log = _session(tmp_path)
  attachment = _attachment(canonical)
  _record_attachment(session_log, attachment)

  async def envelope_reader(
    workflow_run_id: str,
    output_id: str,
    after_char: int,
  ) -> dict[str, object]:
    chunk = canonical[after_char : after_char + 7]
    next_after = after_char + len(chunk)
    end = next_after == len(canonical)
    return {
      "ok": True,
      "action": "output",
      "workflow_run_id": workflow_run_id,
      "output_id": output_id,
      "view": "paged_exact_content",
      "source": attachment.published_output_ref.content.model_dump(
        mode="json"
      ),
      "authorization": {
        "kind": "published_output",
        "workflow_run_id": workflow_run_id,
        "phase_number": attachment.delivery_phase_number,
        "revision": attachment.delivery_revision,
        "output_id": output_id,
      },
      "encoding": "utf-8",
      "after_char": after_char,
      "content": chunk,
      "content_sha256": attachment.content_sha256,
      "content_chars": attachment.content_chars,
      "content_bytes": attachment.content_bytes,
      "next_cursor": None if end else {"after_char": next_after},
      "end": end,
    }

  session.workflow_output_reader = envelope_reader
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
  assert b"workflow_output_integrity_failed" in response.body
  assert [
    record.data["reason_code"]
    for record in caplog.records
    if getattr(record, "data", {}).get("event")
    == "workflow_output_integrity_failed"
  ] == ["invalid_page_shape"]


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
  owner.workflow_output_reader = _published_reader(
    tmp_path, canonical, attachment
  )

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
  session.workflow_output_reader = _lying_page_reader('"tampered!"', attachment)

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
  variant = _variant_source(attachment, **{field_name: field_value})

  async def tampered_reader(
    workflow_run_id: str,
    output_id: str,
    after_char: int,
  ) -> WorkflowContentPage:
    return _content_page(canonical, attachment, after_char, source=variant)

  session.workflow_output_reader = tampered_reader
  response = await workflow_output_response(
    _request(),
    attachment.workflow_run_id,
    attachment.output_id,
    auth=_Auth(session),  # type: ignore[arg-type]
  )

  assert response.status_code == 409
  assert b"workflow_output_integrity_failed" in response.body


def _tampered_source_page(
  content: str,
  attachment: WorkflowOutputAttachment,
  after_char: int,
) -> WorkflowContentPage:
  return _content_page(
    content,
    attachment,
    after_char,
    source=_variant_source(attachment, retention="session"),
  )


def _tampered_authorization_page(
  content: str,
  attachment: WorkflowOutputAttachment,
  after_char: int,
) -> WorkflowContentPage:
  return _content_page(
    content,
    attachment,
    after_char,
    authorization=PublishedOutputPageAuthorization(
      workflow_run_id=attachment.workflow_run_id,
      phase_number=attachment.delivery_phase_number,
      revision=99,
      output_id=attachment.output_id,
    ),
  )


def _tampered_cursor_echo_page(
  content: str,
  attachment: WorkflowOutputAttachment,
  after_char: int,
) -> WorkflowContentPage:
  return _content_page(content, attachment, after_char + 1)


@pytest.mark.parametrize(
  ("expected_reason", "page_factory"),
  [
    ("source_handle_mismatch", _tampered_source_page),
    ("authorization_mismatch", _tampered_authorization_page),
    ("paging_cursor_mismatch", _tampered_cursor_echo_page),
  ],
)
@pytest.mark.asyncio
async def test_each_integrity_variant_logs_its_distinct_internal_reason(
  tmp_path: Path,
  caplog: pytest.LogCaptureFixture,
  expected_reason: str,
  page_factory: object,
) -> None:
  """Internally-valid typed pages claiming a different source, authorization,
  or cursor echo still 409 with their distinct retained reasons; the tamper
  shapes the page validator itself rejects need no route arm."""

  canonical = '"canonical"'
  session, session_log = _session(tmp_path)
  attachment = _attachment(canonical)
  _record_attachment(session_log, attachment)

  async def tampered_reader(
    workflow_run_id: str,
    output_id: str,
    after_char: int,
  ) -> WorkflowContentPage:
    return page_factory(canonical, attachment, after_char)  # type: ignore[operator]

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
async def test_empty_source_page_logs_empty_page_reason(
  tmp_path: Path,
  caplog: pytest.LogCaptureFixture,
) -> None:
  """A non-empty content page over a non-empty source cannot echo the cursor
  and still be empty (the page validator forbids it), so the loop-progress
  guard is reachable only through a genuinely empty published source."""

  session, session_log = _session(tmp_path)
  attachment = _attachment("")
  _record_attachment(session_log, attachment)
  session.workflow_output_reader = _published_reader(tmp_path, "", attachment)

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
  ] == ["empty_page"]


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
  session.workflow_output_reader = _lying_page_reader('"tampered!"', attachment)

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
  session.workflow_output_reader = _published_reader(
    tmp_path, canonical, attachment
  )

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
