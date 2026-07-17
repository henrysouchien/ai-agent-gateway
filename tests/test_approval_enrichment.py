import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
PKG_DIR = ROOT / "packages" / "agent-gateway"
if str(PKG_DIR) not in sys.path:
  sys.path.insert(0, str(PKG_DIR))

from agent_gateway.approval_enrichment import effective_trade_approval_expiry_seconds, enrich_trade_approval_args  # noqa: E402
from agent_gateway.event_log import EventLog  # noqa: E402


def _append_preview(log: EventLog, *, preview_id: str = "pv-1") -> None:
  log.append(
    {
      "type": "tool_call_complete",
      "tool_call_id": "tool-preview",
      "tool_name": "preview_trade",
      "result": {
        "status": "success",
        "metadata": {
          "account_id": "acct-1",
          "expires_at": "2999-01-01T00:00:00",
          "broker_provider": "ibkr",
        },
        "data": {
          "preview_id": preview_id,
          "ticker": "SGOV",
          "side": "BUY",
          "quantity": 12,
          "order_type": "Market",
          "time_in_force": "Day",
          "estimated_price": 100.25,
          "estimated_total": 1203.0,
          "estimated_commission": 0.0,
          "pre_trade_weight": 0.01,
          "post_trade_weight": 0.02,
          "validation": {"is_valid": True, "warnings": ["concentration check"]},
        },
      },
      "error": None,
    }
  )


def test_trade_approval_args_include_matching_preview_summary() -> None:
  log = EventLog()
  _append_preview(log)

  enriched = enrich_trade_approval_args(
    "execute_trade",
    {"preview_id": "pv-1"},
    event_log=log,
  )

  assert enriched["preview_id"] == "pv-1"
  summary = enriched["approval_summary"]
  assert summary["preview_id"] == "pv-1"
  assert summary["ticker"] == "SGOV"
  assert summary["side"] == "BUY"
  assert summary["quantity"] == 12
  assert summary["estimated_total"] == 1203.0
  assert "account_id" not in summary
  assert summary["validation"] == {"is_valid": True, "warnings": ["concentration check"]}


def test_proposal_apply_approval_declares_model_writer_undo_retirement() -> None:
  token_id = f"undo_{'a' * 32}"

  enriched = enrich_trade_approval_args(
    "apply_patch_proposal",
    {
      "proposal_id": "proposal-dcf",
      "confirm_apply": True,
      "source_model_writer_undo_token_id": token_id,
      "source_model_writer_undo_expires_at": 1_800_000_000.0,
      "source_model_writer_undo_effect": "retired_after_apply",
    },
  )

  assert token_id in enriched["consequence"]
  assert "permanently retires" in enriched["consequence"]
  assert enriched["approval_summary"] == {
    "proposal_id": "proposal-dcf",
    "model_writer_undo": {
      "status": "will_be_retired_by_apply",
      "undo_token_id": token_id,
      "undo_expires_at": 1_800_000_000.0,
    },
    "operator_choice": (
      "Approve to promote the Thesis proposal, or deny and use "
      "fms_undo_model_writer_commit before the receipt expires."
    ),
  }


def test_proposal_apply_approval_does_not_invent_undo_retirement() -> None:
  payload = {
    "proposal_id": "proposal-final-review",
    "confirm_apply": True,
  }

  assert enrich_trade_approval_args("apply_patch_proposal", payload) == payload


def test_trade_approval_args_do_not_cross_preview_ids() -> None:
  log = EventLog()
  _append_preview(log, preview_id="pv-other")

  assert enrich_trade_approval_args(
    "execute_trade",
    {
      "preview_id": "pv-1",
      "approval_summary": {"ticker": "SPOOFED"},
      "approvalSummary": {"ticker": "SPOOFED"},
      "preview_summary": {"ticker": "SPOOFED"},
      "previewSummary": {"ticker": "SPOOFED"},
      "trade_preview": {"ticker": "SPOOFED"},
      "tradePreview": {"ticker": "SPOOFED"},
    },
    event_log=log,
  ) == {"preview_id": "pv-1"}


def test_trade_approval_args_overwrite_untrusted_existing_summary() -> None:
  log = EventLog()
  _append_preview(log)

  assert enrich_trade_approval_args(
    "mcp__portfolio-trades-mcp__execute_trade",
    {"preview_id": "pv-1", "approval_summary": {"ticker": "USER"}},
    event_log=log,
  )["approval_summary"]["ticker"] == "SGOV"


def test_trade_approval_args_do_not_attach_expired_preview_summary() -> None:
  log = EventLog()
  log.append(
    {
      "type": "tool_call_complete",
      "tool_call_id": "tool-preview",
      "tool_name": "preview_trade",
      "result": {
        "status": "success",
        "metadata": {"expires_at": "2000-01-01T00:00:00"},
        "data": {
          "preview_id": "pv-1",
          "ticker": "SGOV",
          "side": "BUY",
          "quantity": 12,
          "estimated_total": 1203.0,
        },
      },
      "error": None,
    }
  )

  assert enrich_trade_approval_args(
    "execute_trade",
    {"preview_id": "pv-1"},
    event_log=log,
  ) == {"preview_id": "pv-1"}


def test_basket_trade_approval_args_include_matching_preview_summary() -> None:
  log = EventLog()
  log.append(
    {
      "type": "tool_call_complete",
      "tool_call_id": "tool-preview",
      "tool_name": "preview_basket_trade",
      "result": {
        "status": "success",
        "basket_name": "quality",
        "action": "buy",
        "preview_ids": ["pv-aapl", "pv-msft"],
        "total_estimated_cost": 801.0,
        "gross_estimated_notional": 800.0,
        "preview_legs": [
          {
            "ticker": "AAPL",
            "side": "BUY",
            "quantity": 2,
            "estimated_price": 150.0,
            "estimated_total": 301.0,
            "preview_id": "pv-aapl",
            "status": "success",
            "expires_at": "2999-01-01T00:00:00",
          },
          {
            "ticker": "MSFT",
            "side": "BUY",
            "quantity": 1,
            "estimated_price": 500.0,
            "estimated_total": 500.0,
            "preview_id": "pv-msft",
            "status": "success",
            "expires_at": "2999-01-01T00:00:00",
          },
        ],
      },
      "error": None,
    }
  )

  enriched = enrich_trade_approval_args(
    "execute_basket_trade",
    {"preview_ids": '["pv-aapl","pv-msft"]'},
    event_log=log,
  )

  summary = enriched["approval_summary"]
  assert summary["preview_ids"] == ["pv-aapl", "pv-msft"]
  assert summary["expires_at"] == "2999-01-01T00:00:00+00:00"
  assert summary["basket_name"] == "quality"
  assert summary["total_estimated_cost"] == 801.0
  assert summary["gross_estimated_notional"] == 800.0
  assert summary["legs"][0]["ticker"] == "AAPL"
  assert summary["legs"][0]["estimated_total"] == 301.0


def test_basket_trade_approval_args_require_exact_preview_id_set() -> None:
  log = EventLog()
  log.append(
    {
      "type": "tool_call_complete",
      "tool_call_id": "tool-preview",
      "tool_name": "preview_basket_trade",
      "result": {
        "status": "success",
        "snapshot": {
          "preview_ids": ["pv-aapl", "pv-msft"],
          "basket_name": "quality",
        },
      },
      "error": None,
    }
  )

  assert enrich_trade_approval_args(
    "execute_basket_trade",
    {"preview_ids": ["pv-aapl"]},
    event_log=log,
  ) == {"preview_ids": ["pv-aapl"]}


def test_basket_trade_approval_args_use_earliest_mixed_timezone_leg_expiry() -> None:
  log = EventLog()
  log.append(
    {
      "type": "tool_call_complete",
      "tool_call_id": "tool-preview",
      "tool_name": "preview_basket_trade",
      "result": {
        "status": "success",
        "preview_ids": ["pv-aapl", "pv-msft"],
        "preview_legs": [
          {
            "ticker": "AAPL",
            "side": "BUY",
            "quantity": 2,
            "preview_id": "pv-aapl",
            "status": "success",
            "expires_at": "2999-01-01T00:01:00",
          },
          {
            "ticker": "MSFT",
            "side": "BUY",
            "quantity": 1,
            "preview_id": "pv-msft",
            "status": "success",
            "expires_at": "2999-01-01T00:02:00+00:00",
          },
        ],
      },
      "error": None,
    }
  )

  enriched = enrich_trade_approval_args(
    "execute_basket_trade",
    {"preview_ids": ["pv-aapl", "pv-msft"]},
    event_log=log,
  )

  assert enriched["approval_summary"]["expires_at"] == "2999-01-01T00:01:00+00:00"


def test_trade_approval_expiry_caps_to_preview_remaining_with_dispatch_margin() -> None:
  now = datetime(2026, 6, 15, 12, 0, tzinfo=UTC)
  expires_at = (now + timedelta(seconds=100)).isoformat()

  assert effective_trade_approval_expiry_seconds(
    "execute_trade",
    {"approval_summary": {"expires_at": expires_at}},
    requested_expiry_seconds=600,
    max_wait_seconds=270,
    now=now,
  ) == 70.0


def test_trade_approval_expiry_treats_naive_preview_expiry_as_utc() -> None:
  now = datetime(2026, 6, 15, 12, 0, tzinfo=UTC)

  assert effective_trade_approval_expiry_seconds(
    "execute_trade",
    {"approval_summary": {"expires_at": "2026-06-15T12:01:40"}},
    requested_expiry_seconds=600,
    max_wait_seconds=270,
    now=now,
  ) == 70.0


def test_trade_approval_expiry_uses_global_wait_when_preview_lives_longer() -> None:
  now = datetime(2026, 6, 15, 12, 0, tzinfo=UTC)
  expires_at = (now + timedelta(seconds=900)).isoformat()

  assert effective_trade_approval_expiry_seconds(
    "execute_trade",
    {"approval_summary": {"expires_at": expires_at}},
    requested_expiry_seconds=600,
    max_wait_seconds=270,
    now=now,
  ) == 270.0


def test_trade_approval_expiry_fails_nearly_closed_when_preview_margin_is_gone() -> None:
  now = datetime(2026, 6, 15, 12, 0, tzinfo=UTC)
  expires_at = (now + timedelta(seconds=20)).isoformat()

  assert effective_trade_approval_expiry_seconds(
    "execute_trade",
    {"approval_summary": {"expires_at": expires_at}},
    requested_expiry_seconds=600,
    max_wait_seconds=270,
    now=now,
  ) == 0.1


def test_trade_approval_expiry_does_not_change_non_trade_tools() -> None:
  now = datetime(2026, 6, 15, 12, 0, tzinfo=UTC)

  assert effective_trade_approval_expiry_seconds(
    "memory_write",
    {"approval_summary": {"expires_at": (now + timedelta(seconds=20)).isoformat()}},
    requested_expiry_seconds=600,
    max_wait_seconds=270,
    now=now,
  ) == 600.0


def test_basket_trade_approval_args_require_leg_expiry() -> None:
  log = EventLog()
  log.append(
    {
      "type": "tool_call_complete",
      "tool_call_id": "tool-preview",
      "tool_name": "preview_basket_trade",
      "result": {
        "status": "success",
        "preview_ids": ["pv-aapl"],
        "preview_legs": [
          {
            "ticker": "AAPL",
            "side": "BUY",
            "quantity": 2,
            "preview_id": "pv-aapl",
            "status": "success",
          }
        ],
      },
      "error": None,
    }
  )

  assert enrich_trade_approval_args(
    "execute_basket_trade",
    {"preview_ids": ["pv-aapl"]},
    event_log=log,
  ) == {"preview_ids": ["pv-aapl"]}
