from __future__ import annotations

import asyncio
from types import SimpleNamespace
from uuid import uuid4

from agent_gateway import ApprovalDecision, ToolDispatcher


class _Mcp:
  def __init__(self, sequence=None) -> None:
    self.calls = []
    self.sequence = sequence

  def is_mcp_tool(self, _name: str) -> bool:
    return True

  def get_server_for_tool(self, _name: str) -> str:
    return "portfolio-trades-mcp"

  async def call_tool(self, name, tool_input, **kwargs):
    if self.sequence is not None:
      self.sequence.append("mcp")
    self.calls.append((name, tool_input, kwargs))
    return {"ok": True}, None


def _commercial():
  context_id = uuid4()
  authorization_id = uuid4()
  workflow_id = uuid4()
  return SimpleNamespace(
    claim=SimpleNamespace(context_id=context_id, entitlement_revision=42),
    authorization=SimpleNamespace(
      authorization_id=authorization_id,
      workflow_run_id=workflow_id,
      request_id="request-commercial-1",
      session_id="session-commercial-1",
      operation="messages.create",
      capability_id="portfolio.review",
      provider="anthropic",
      billing_mode="metered",
    ),
  )


def test_irreversible_rechecks_after_approval_and_injects_token_free_lineage() -> None:
  sequence = []
  mcp = _Mcp(sequence)
  commercial = _commercial()
  rechecks = []

  async def approve(_request):
    sequence.append("approval")
    return ApprovalDecision(approved=True)

  def recheck(context):
    sequence.append("recheck")
    rechecks.append(context)

  dispatcher = ToolDispatcher(
    mcp_client=mcp,
    role="owner",
    needs_approval=lambda *_args, **_kwargs: True,
    request_approval=approve,
    commercial_work_start=commercial,
    commercial_irreversible_recheck=recheck,
    commercial_mcp_servers=frozenset({"portfolio-trades-mcp"}),
  )

  result, error = asyncio.run(
    dispatcher.dispatch(
      "call-1",
      "execute_trade",
      {"preview_id": "preview-1"},
      advertised_tool_names=frozenset({"execute_trade"}),
    )
  )

  assert error is None
  assert result == {"ok": True}
  assert rechecks == [commercial]
  assert sequence == ["approval", "recheck", "mcp"]
  meta = mcp.calls[0][2]["meta"]["hank_commercial"]
  assert meta == {
    "tool_name": "execute_trade",
    "execution_context_id": str(commercial.claim.context_id),
    "work_authorization_id": str(commercial.authorization.authorization_id),
    "workflow_run_id": str(commercial.authorization.workflow_run_id),
    "entitlement_revision": 42,
    "request_id": "request-commercial-1",
    "session_id": "session-commercial-1",
    "operation": "messages.create",
    "capability_id": "portfolio.review",
    "provider": "anthropic",
    "billing_mode": "metered",
  }


def test_irreversible_live_recheck_failure_never_calls_mcp() -> None:
  mcp = _Mcp()

  def deny(_context):
    raise RuntimeError("revoked")

  dispatcher = ToolDispatcher(
    mcp_client=mcp,
    role="owner",
    needs_approval=lambda *_args, **_kwargs: False,
    commercial_work_start=_commercial(),
    commercial_irreversible_recheck=deny,
    commercial_mcp_servers=frozenset({"portfolio-trades-mcp"}),
  )

  result, error = asyncio.run(
    dispatcher.dispatch(
      "call-1",
      "execute_trade",
      {"preview_id": "preview-1"},
      advertised_tool_names=frozenset({"execute_trade"}),
    )
  )

  assert result is None
  assert error == {
    "code": "commercial_irreversible_authority_invalid",
    "message": "Fresh commercial authority is invalid or expired.",
  }
  assert mcp.calls == []


def test_commercial_dispatch_denies_untrusted_mcp_without_metadata_leak() -> None:
  mcp = _Mcp()
  dispatcher = ToolDispatcher(
    mcp_client=mcp,
    role="owner",
    needs_approval=lambda *_args, **_kwargs: False,
    commercial_work_start=_commercial(),
    commercial_irreversible_recheck=lambda _context: None,
    commercial_mcp_servers=frozenset({"different-server"}),
  )

  result, error = asyncio.run(
    dispatcher.dispatch(
      "call-1",
      "execute_trade",
      {"preview_id": "preview-1"},
      advertised_tool_names=frozenset({"execute_trade"}),
    )
  )

  assert result is None
  assert error["code"] == "commercial_mcp_destination_denied"
  assert mcp.calls == []


def test_missing_irreversible_recheck_fails_closed() -> None:
  mcp = _Mcp()
  dispatcher = ToolDispatcher(
    mcp_client=mcp,
    role="owner",
    needs_approval=lambda *_args, **_kwargs: False,
    commercial_work_start=_commercial(),
    commercial_mcp_servers=frozenset({"portfolio-trades-mcp"}),
  )

  result, error = asyncio.run(
    dispatcher.dispatch(
      "call-1",
      "execute_trade",
      {"preview_id": "preview-1"},
      advertised_tool_names=frozenset({"execute_trade"}),
    )
  )

  assert result is None
  assert error["code"] == "commercial_irreversible_authority_unavailable"
  assert mcp.calls == []


def test_noncommercial_dispatch_remains_metadata_free() -> None:
  mcp = _Mcp()
  dispatcher = ToolDispatcher(
    mcp_client=mcp,
    role="owner",
    needs_approval=lambda *_args, **_kwargs: False,
  )

  result, error = asyncio.run(
    dispatcher.dispatch(
      "call-1",
      "execute_trade",
      {"preview_id": "preview-1"},
      advertised_tool_names=frozenset({"execute_trade"}),
    )
  )

  assert error is None
  assert result == {"ok": True}
  assert "meta" not in mcp.calls[0][2]
