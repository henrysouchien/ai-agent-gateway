# ruff: noqa: E402

from __future__ import annotations

import asyncio
import logging
import sys
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[3]
PKG_DIR = ROOT / "packages" / "agent-gateway"
if str(PKG_DIR) not in sys.path:
  sys.path.insert(0, str(PKG_DIR))

from agent_gateway import EventLog, ToolDispatcher
from agent_gateway import policy_imports
from agent_gateway import tool_dispatcher as dispatcher_module
from agent_gateway import tool_dispatcher_approval_lifecycle as lifecycle_helpers
from agent_gateway.approval_policy import (
  ApprovalDecision as PolicyApprovalDecision,
  PersistentGrant,
  RunContext,
  build_approval_request,
  utc_now,
)
from agent_gateway.approval_store import SQLiteApprovalStore
from agent_gateway.prepared_business_model_store import PreparedBusinessModelLifecycle
from agent_gateway.batch_approval_projection import (
  BatchApprovalProjectionRegistry,
  BatchApprovalScope,
)
from agent_gateway.single_user_policy import SingleUserApprovalPolicy
from agent_gateway.tool_dispatcher_helpers import PlannedWritePlanningRejected
from api.fms.core.change_set import (
  ArtifactOnlyPlan,
  ArtifactPayload,
  BaseRevision,
  CanonicalPayload,
  ChangeSet,
  CommitStrategy,
  DomainResultRef,
  EffectCriticality,
  EffectKind,
  EffectSpec,
  InlinePayload,
  IntentRef,
  JournaledCasPlan,
  ProducerRef,
  ReviewKind,
  ReviewRequirement,
  SnapshotPayload,
  StoreTarget,
  StoreWritePayload,
  TargetRef,
  TargetScope,
  compute_legacy_journal_base_vector_hash,
)
from schema.cas import journal_content_id


class _NullMcp:
  def is_mcp_tool(self, _name: str) -> bool:
    return False

  def get_server_for_tool(self, _name: str) -> str | None:
    return None

  async def call_tool(self, _name: str, _tool_input: dict[str, Any], **_kwargs: Any):
    return {"ok": True}, None


class _PrefixedMcp(_NullMcp):
  def is_mcp_tool(self, name: str) -> bool:
    return name == "trades_execute_trade"

  def get_server_for_tool(self, name: str) -> str | None:
    return "portfolio-trades-mcp" if name == "trades_execute_trade" else None

  def get_original_tool_name(self, name: str) -> str:
    return "execute_trade" if name == "trades_execute_trade" else name


class _NoMcpLookup:
  async def call_tool(self, _name: str, _tool_input: dict[str, Any], **_kwargs: Any):
    return {"ok": True}, None


class _SessionLog:
  def __init__(self) -> None:
    self.events: list[dict[str, Any]] = []

  async def append(self, event: dict[str, Any]) -> None:
    self.events.append(dict(event))


def _request() -> SimpleNamespace:
  return SimpleNamespace(
    approval_id="approval-1",
    tool_call_id="call-1",
    tool_name="place_order",
    tool_args_redacted={"ticker": "MSFT"},
  )


def _decision() -> SimpleNamespace:
  return SimpleNamespace(
    reason="needs approval",
    allow_persistent_grant=True,
  )


def test_tool_dispatcher_resolve_tool_class_uses_policy_owner(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  policy_module = SimpleNamespace(
    get_forbidden_tools_for_session=lambda _session: frozenset({"execute_trade"}),
    get_server_for_policy_tool=lambda tool_name: "portfolio-trades-mcp"
    if tool_name == "execute_trade"
    else None,
    get_tool_class=lambda server_name, tool_name: "irreversible"
    if (server_name, tool_name) == ("portfolio-trades-mcp", "execute_trade")
    else None,
  )

  monkeypatch.setattr(
    policy_imports.importlib,
    "import_module",
    lambda name: policy_module if name == "agent.shared.server_policies" else None,
  )

  dispatcher = ToolDispatcher(mcp_client=_NullMcp(), local_tool_handlers={}, event_log=EventLog())

  assert dispatcher._resolve_tool_class("execute_trade") == "irreversible"


def test_tool_dispatcher_resolve_tool_class_preserves_parent_policy_resolver_seam(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  calls: list[dict[str, Any]] = []

  def fake_resolve_server_policy_tool_class(tool_name: str, **kwargs: Any) -> str:
    calls.append({"tool_name": tool_name, **kwargs})
    return "artifact_write"

  monkeypatch.setattr(
    dispatcher_module,
    "resolve_server_policy_tool_class",
    fake_resolve_server_policy_tool_class,
  )
  dispatcher = ToolDispatcher(mcp_client=_NullMcp(), local_tool_handlers={}, event_log=EventLog())

  assert dispatcher._resolve_tool_class("emit_artifact") == "artifact_write"
  assert calls == [
    {
      "tool_name": "emit_artifact",
      "policy_tool_name": "emit_artifact",
      "runtime_server": None,
      "default": "",
    }
  ]


def test_tool_dispatcher_resolve_tool_class_does_not_require_mcp_lookup_methods(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  policy_module = SimpleNamespace(
    get_forbidden_tools_for_session=lambda _session: frozenset({"record_workflow_action"}),
    get_server_for_policy_tool=lambda tool_name: "portfolio-reads-mcp"
    if tool_name == "record_workflow_action"
    else None,
    get_tool_class=lambda server_name, tool_name: "state_write"
    if (server_name, tool_name) == ("portfolio-reads-mcp", "record_workflow_action")
    else None,
  )

  monkeypatch.setattr(
    policy_imports.importlib,
    "import_module",
    lambda name: policy_module if name == "agent.shared.server_policies" else None,
  )

  dispatcher = ToolDispatcher(mcp_client=_NoMcpLookup(), local_tool_handlers={}, event_log=EventLog())

  assert dispatcher._resolve_tool_class("record_workflow_action") == "state_write"


def test_tool_dispatcher_resolve_tool_class_uses_original_name_for_prefixed_mcp_tool(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  policy_module = SimpleNamespace(
    get_forbidden_tools_for_session=lambda _session: frozenset({"execute_trade"}),
    get_server_for_policy_tool=lambda tool_name: "portfolio-trades-mcp"
    if tool_name == "execute_trade"
    else None,
    get_tool_class=lambda server_name, tool_name: "irreversible"
    if (server_name, tool_name) == ("portfolio-trades-mcp", "execute_trade")
    else None,
  )

  monkeypatch.setattr(
    policy_imports.importlib,
    "import_module",
    lambda name: policy_module if name == "agent.shared.server_policies" else None,
  )

  dispatcher = ToolDispatcher(mcp_client=_PrefixedMcp(), local_tool_handlers={}, event_log=EventLog())

  assert dispatcher._resolve_tool_class("trades_execute_trade") == "irreversible"


def test_tool_dispatcher_resolve_tool_class_uses_policy_owner_after_split_runtime_server(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  policy_module = SimpleNamespace(
    get_forbidden_tools_for_session=lambda _session: frozenset({"execute_trade"}),
    get_server_for_policy_tool=lambda tool_name: "portfolio-trades-mcp"
    if tool_name == "execute_trade"
    else None,
    get_tool_class=lambda server_name, tool_name: "irreversible"
    if (server_name, tool_name) == ("portfolio-trades-mcp", "execute_trade")
    else None,
  )

  monkeypatch.setattr(
    policy_imports.importlib,
    "import_module",
    lambda name: policy_module if name == "agent.shared.server_policies" else None,
  )

  dispatcher = ToolDispatcher(mcp_client=_PrefixedMcp(), local_tool_handlers={}, event_log=EventLog())

  assert dispatcher._resolve_tool_class("trades_execute_trade") == "irreversible"


def test_tool_dispatcher_resolve_tool_class_falls_back_when_policy_modules_absent(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  def fake_import_module(name: str) -> Any:
    if name == "agent.shared.server_policies":
      raise ModuleNotFoundError("No module named 'agent'", name="agent")
    if name == "api.agent.shared.server_policies":
      raise ModuleNotFoundError("No module named 'api'", name="api")
    raise AssertionError(f"unexpected import: {name}")

  monkeypatch.setattr(policy_imports.importlib, "import_module", fake_import_module)
  dispatcher = ToolDispatcher(mcp_client=_NullMcp(), local_tool_handlers={}, event_log=EventLog())

  assert dispatcher._resolve_tool_class("unmapped_tool") == "state_write"


def test_tool_dispatcher_resolve_tool_class_raises_when_policy_import_dependency_breaks(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  def fake_import_module(_name: str) -> Any:
    raise ModuleNotFoundError("No module named 'broken_dependency'", name="broken_dependency")

  monkeypatch.setattr(policy_imports.importlib, "import_module", fake_import_module)
  dispatcher = ToolDispatcher(mcp_client=_NullMcp(), local_tool_handlers={}, event_log=EventLog())

  with pytest.raises(ModuleNotFoundError, match="broken_dependency"):
    dispatcher._resolve_tool_class("execute_trade")


def test_tool_dispatcher_resolve_tool_class_raises_when_policy_module_lacks_class_helper(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  policy_module = SimpleNamespace(
    get_forbidden_tools_for_session=lambda _session: frozenset({"execute_trade"}),
    get_server_for_policy_tool=lambda tool_name: "portfolio-reads-mcp"
    if tool_name == "execute_trade"
    else None,
  )

  monkeypatch.setattr(
    policy_imports.importlib,
    "import_module",
    lambda name: policy_module if name == "agent.shared.server_policies" else None,
  )
  dispatcher = ToolDispatcher(mcp_client=_NullMcp(), local_tool_handlers={}, event_log=EventLog())

  with pytest.raises(AttributeError, match="get_tool_class"):
    dispatcher._resolve_tool_class("execute_trade")


async def _wait_for_queue(session: SimpleNamespace, tool_call_id: str) -> asyncio.Queue:
  for _ in range(100):
    queue = session.approval_queues.get(tool_call_id)
    if queue is not None:
      return queue
    await asyncio.sleep(0)
  raise AssertionError("approval queue was not registered")


def test_projected_batch_approval_waits_through_its_advertised_expiry() -> None:
  calls: list[float | int | None] = []

  def globally_capped(expiry_seconds: float | int | None) -> float:
    calls.append(expiry_seconds)
    return 270.0

  assert lifecycle_helpers._approval_wait_timeout_seconds(
    600,
    batch_admission=object(),
    approval_queue_timeout_seconds_fn=globally_capped,
  ) == 600.0
  assert calls == []


def test_non_batch_approval_keeps_global_wait_ceiling() -> None:
  assert lifecycle_helpers._approval_wait_timeout_seconds(
    600,
    batch_admission=None,
    approval_queue_timeout_seconds_fn=lambda _expiry: 270.0,
  ) == 270.0


def test_pending_tool_helper_registers_event_waits_and_cleans_up() -> None:
  async def scenario() -> None:
    session_log = _SessionLog()
    session = SimpleNamespace(pending_tools={}, approval_queues={}, agent_session_log=session_log)
    event_log = EventLog()
    request = _request()
    task = asyncio.create_task(
      lifecycle_helpers.await_user_approval_via_pending_tools(
        session=session,
        approval_store=None,
        event_log=event_log,
        request=request,
        decision=_decision(),
        nonce="nonce-1",
        resolved_qualifier="qual-1",
        allow_persistent=True,
        timeout_seconds=5,
        log=logging.getLogger("test"),
      )
    )

    queue = await _wait_for_queue(session, request.tool_call_id)
    assert session.pending_tools[request.tool_call_id] == {
      "approval_id": "approval-1",
      "nonce": "nonce-1",
      "requested_at": session.pending_tools[request.tool_call_id]["requested_at"],
      "status": "approval_pending",
      "tool_name": "place_order",
      "resolved_qualifier": "qual-1",
    }
    assert event_log.entries[0].event["type"] == "tool_approval_request"
    assert session_log.events[0]["allow_persistent_approval"] is True

    await queue.put({"approved": True, "allow_tool_type": False})

    assert await task == {"approved": True, "allow_tool_type": False}
    assert session.pending_tools == {}
    assert session.approval_queues == {}

  asyncio.run(scenario())


def test_projected_pending_tool_binds_stage_identity_to_projection_and_event() -> None:
  async def scenario() -> None:
    session_log = _SessionLog()
    session = SimpleNamespace(
      pending_tools={},
      approval_queues={},
      agent_session_log=session_log,
      batch_stage_run_seq=3,
    )
    event_log = EventLog()
    request = _request()

    class Admission:
      def publish_pending(self) -> None:
        session.approval_queues[request.tool_call_id].put_nowait({"approved": False})

    result = await lifecycle_helpers.await_user_approval_via_pending_tools(
      session=session,
      approval_store=None,
      event_log=event_log,
      request=request,
      decision=_decision(),
      nonce="nonce-1",
      resolved_qualifier="qual-1",
      allow_persistent=False,
      timeout_seconds=5,
      log=logging.getLogger("test"),
      batch_admission=Admission(),
    )

    assert result == {"approved": False}
    assert event_log.entries[0].event["stage_run_seq"] == 3
    assert session_log.events[0]["stage_run_seq"] == 3
    assert session.pending_tools == {}
    assert session.approval_queues == {}

  asyncio.run(scenario())


@pytest.mark.parametrize("stage_run_seq", [None, 0, -1, True, "3"])
def test_projected_pending_tool_rejects_invalid_stage_identity(
  stage_run_seq: object,
) -> None:
  session = SimpleNamespace(
    pending_tools={},
    approval_queues={},
    batch_stage_run_seq=stage_run_seq,
  )

  with pytest.raises(
    ValueError,
    match="stage_run_seq must be a positive integer",
  ):
    asyncio.run(
      lifecycle_helpers.await_user_approval_via_pending_tools(
        session=session,
        approval_store=None,
        event_log=None,
        request=_request(),
        decision=_decision(),
        nonce="nonce-1",
        resolved_qualifier="",
        allow_persistent=False,
        timeout_seconds=5,
        log=logging.getLogger("test"),
        batch_admission=object(),
      )
    )
  assert session.pending_tools == {}
  assert session.approval_queues == {}


def test_pending_tool_helper_expires_store_request_on_timeout() -> None:
  class Store:
    def __init__(self) -> None:
      self.transitions: list[dict[str, Any]] = []

    async def get(self, approval_id: str) -> SimpleNamespace:
      assert approval_id == "approval-1"
      return SimpleNamespace(approval_id=approval_id, state="pending_user", state_version=7)

    async def transition_state(self, approval_id: str, state: str, **kwargs: Any) -> SimpleNamespace:
      self.transitions.append({"approval_id": approval_id, "state": state, **kwargs})
      return SimpleNamespace(approval_id=approval_id, state=state, state_version=8)

  async def scenario() -> None:
    store = Store()
    session = SimpleNamespace(pending_tools={}, approval_queues={}, agent_session_log=None)

    result = await lifecycle_helpers.await_user_approval_via_pending_tools(
      session=session,
      approval_store=store,
      event_log=None,
      request=_request(),
      decision=_decision(),
      nonce="nonce-1",
      resolved_qualifier="",
      allow_persistent=False,
      timeout_seconds=0,
      log=logging.getLogger("test"),
    )

    assert result is None
    assert store.transitions == [
      {
        "approval_id": "approval-1",
        "state": "expired",
        "expected_state_version": 7,
        "decision_reason": "Timed out waiting for user approval",
      }
    ]
    assert session.pending_tools == {}
    assert session.approval_queues == {}

  asyncio.run(scenario())


def test_tool_dispatcher_pending_approval_wrapper_threads_instance_state(monkeypatch) -> None:
  captured: dict[str, Any] = {}

  async def fake_await_user_approval_via_pending_tools(**kwargs: Any) -> dict[str, bool]:
    captured.update(kwargs)
    return {"approved": True}

  monkeypatch.setattr(
    lifecycle_helpers,
    "await_user_approval_via_pending_tools",
    fake_await_user_approval_via_pending_tools,
  )
  session = SimpleNamespace(
    pending_tools={},
    approval_queues={},
    approval_store="store",
    approval_policy="policy",
  )
  dispatcher = ToolDispatcher(
    mcp_client=_NullMcp(),
    local_tool_handlers={},
    event_log=EventLog(),
    session=session,
  )
  request = _request()
  decision = _decision()

  result = asyncio.run(
    dispatcher._await_user_approval_via_pending_tools(
      request,
      decision,
      nonce="nonce-1",
      resolved_qualifier="qual-1",
      allow_persistent=True,
      timeout_seconds=15,
    )
  )

  assert result == {"approved": True}
  assert captured["session"] is session
  assert captured["approval_store"] == "store"
  assert captured["event_log"] is dispatcher._event_log
  assert captured["request"] is request
  assert captured["decision"] is decision
  assert captured["nonce"] == "nonce-1"
  assert captured["resolved_qualifier"] == "qual-1"
  assert captured["allow_persistent"] is True
  assert captured["timeout_seconds"] == 15


def test_tool_dispatcher_run_approval_lifecycle_wrapper_threads_instance_state(monkeypatch) -> None:
  captured: dict[str, Any] = {}

  async def fake_run_approval_lifecycle(**kwargs: Any) -> dict[str, bool]:
    captured.update(kwargs)
    return {"approved": True}

  monkeypatch.setattr(
    lifecycle_helpers,
    "run_approval_lifecycle",
    fake_run_approval_lifecycle,
  )
  session = SimpleNamespace(
    pending_tools={},
    approval_queues={},
    approval_store="store",
    approval_policy="policy",
  )
  dispatcher = ToolDispatcher(
    mcp_client=_NullMcp(),
    local_tool_handlers={},
    event_log=EventLog(),
    session=session,
  )

  result = asyncio.run(
    dispatcher._run_approval_lifecycle(
      tool_call_id="call-1",
      tool_name="place_order",
      tool_input={"ticker": "MSFT"},
      qualifier="qual-1",
      reason="needs approval",
      allow_persistent=True,
    )
  )

  assert result == {"approved": True}
  assert captured["store"] == "store"
  assert captured["policy"] == "policy"
  assert captured["session"] is session
  assert captured["tool_call_id"] == "call-1"
  assert captured["tool_name"] == "place_order"
  assert captured["tool_input"] == {"ticker": "MSFT"}
  assert captured["qualifier"] == "qual-1"
  assert captured["reason"] == "needs approval"
  assert captured["allow_persistent"] is True
  assert captured["resolve_run_context_fn"].__self__ is dispatcher
  assert captured["redact_for_approval_request_fn"].__self__ is dispatcher
  assert captured["resolve_tool_class_fn"].__self__ is dispatcher
  assert captured["effective_trade_approval_decision_fn"].__self__ is dispatcher
  assert captured["await_user_approval_via_pending_tools_fn"].__self__ is dispatcher
  assert callable(captured["current_skill_fn"])
  assert callable(captured["approval_queue_timeout_seconds_fn"])


def _planned_change_set() -> ChangeSet:
  digest = "a" * 64

  def canonical(value: object) -> CanonicalPayload:
    return CanonicalPayload.from_value(value)

  def inline(value: object) -> InlinePayload:
    return InlinePayload("v1", "application/json", canonical(value).content)

  artifact_path = "artifacts/TEST/planned.json"
  effect = EffectSpec(
    "artifact",
    EffectKind.ARTIFACT_REFUSAL_ONLY,
    EffectCriticality.REQUIRED,
    (),
    ArtifactPayload(
      artifact_path,
      canonical({"status": "planned"}),
      inline({"status": "planned"}),
    ),
  )
  return ChangeSet(
    "v1",
    "",
    "",
    ProducerRef("test", "research_producer", "alice", "run-1"),
    TargetRef("TEST", 7, "workspace/TEST", TargetScope.WORKSPACE),
    (BaseRevision("workbook", "models/TEST.xlsx", digest),),
    IntentRef("planned_write", canonical({"x": 1})),
    DomainResultRef("test", "v1", inline({"ok": True})),
    (effect,),
    (),
    ReviewRequirement(ReviewKind.NONE, None),
    CommitStrategy.ARTIFACT_ONLY,
    ArtifactOnlyPlan(artifact_path),
  )


def _planned_workbook_change_set(
  subcommand: str = "persist_dcf_relative_valuation",
) -> ChangeSet:
  base_hash = f"{'a' * 64}:engine-v1"
  target_hash = f"{'b' * 64}:engine-v1"
  projection_base_hash = "c" * 64
  projection_target_hash = "d" * 64
  operations = [
    {
      "type": "set_value",
      "item_id": "tpl.a.unit_economics.gross_margin",
      "values": {"2027": 0.42, "2028": 0.43},
    }
  ]
  effect_rows = [
    EffectSpec(
      "snapshot-workbook",
      EffectKind.SNAPSHOT_WORKBOOK,
      EffectCriticality.REQUIRED,
      (),
      SnapshotPayload(
        "workbook",
        "models/PCTY.xlsx",
        base_hash,
        InlinePayload(
          "v1",
          "application/json",
          CanonicalPayload.from_value({"target": "models/PCTY.xlsx"}).content,
        ),
      ),
    ),
    EffectSpec(
      "write-workbook",
      EffectKind.WRITE_WORKBOOK_BUNDLE,
      EffectCriticality.REQUIRED,
      ("snapshot-workbook",),
      StoreWritePayload(
        "workbook",
        "models/PCTY.xlsx",
        base_hash,
        target_hash,
        CanonicalPayload.from_value({"target_hash": target_hash}),
        InlinePayload(
          "v1",
          "application/json",
          CanonicalPayload.from_value(
            {
              "operations": operations,
              "expected_readback": {"gross_margin": {"2027": 0.42}},
              "force_overwrite": True,
            }
          ).content,
        ),
      ),
    ),
  ]
  base_vector = [BaseRevision("workbook", "models/PCTY.xlsx", base_hash)]
  target_hashes = {"workbook": target_hash}
  target_keys = ["models/PCTY.xlsx"]
  forecast_mode = None
  if subcommand == "persist_forecast_assumptions":
    projection_key = "overrides/PCTY.json"
    forecast_mode = "main_write"
    effect_rows.extend(
      [
        EffectSpec(
          "snapshot-projection",
          EffectKind.SNAPSHOT_OVERRIDES,
          EffectCriticality.REQUIRED,
          (),
          SnapshotPayload(
            "projection_override",
            projection_key,
            projection_base_hash,
            InlinePayload(
              "v1",
              "application/json",
              CanonicalPayload.from_value({"target": projection_key}).content,
            ),
          ),
        ),
        EffectSpec(
          "write-projection",
          EffectKind.WRITE_OVERRIDE_JSON,
          EffectCriticality.REQUIRED,
          ("snapshot-projection",),
          StoreWritePayload(
            "projection_override",
            projection_key,
            projection_base_hash,
            projection_target_hash,
            CanonicalPayload.from_value({"ticker": "PCTY"}),
            InlinePayload(
              "v1",
              "application/json",
              CanonicalPayload.from_value({"ticker": "PCTY", "drivers": {}}).content,
            ),
          ),
        ),
      ]
    )
    base_vector.append(
      BaseRevision("projection_override", projection_key, projection_base_hash)
    )
    target_hashes["projection_override"] = projection_target_hash
    target_keys.insert(0, projection_key)
  effects = tuple(effect_rows)
  target_keys_tuple = tuple(target_keys)
  store_targets = tuple(
    StoreTarget(store_id, target_hashes[store_id])
    for store_id in sorted(target_hashes)
  )
  strategy_plan = JournaledCasPlan(
    content_id=journal_content_id(
      subcommand,
      target_store_hashes=target_hashes,
      target_keys=target_keys_tuple,
      forecast_mode=forecast_mode,
    ),
    legacy_base_vector_hash=compute_legacy_journal_base_vector_hash(effects),
    subcommand=subcommand,
    target_store_hashes=store_targets,
    target_keys=target_keys_tuple,
    forecast_mode=forecast_mode,
  )
  return ChangeSet(
    "v1",
    "",
    "",
    ProducerRef("dcf-relative-valuation", "research_producer", "alice", "run-1"),
    TargetRef("PCTY", 7, "workspace/PCTY", TargetScope.WORKSPACE),
    tuple(sorted(base_vector, key=lambda base: base.store_id)),
    IntentRef(
      subcommand,
      CanonicalPayload.from_value({"ticker": "PCTY"}),
    ),
    DomainResultRef(
      "dcf_relative_valuation",
      "v1",
      InlinePayload(
        "v1",
        "application/json",
        CanonicalPayload.from_value({"ok": True}).content,
      ),
    ),
    effects,
    (),
    ReviewRequirement(ReviewKind.NONE, None),
    CommitStrategy.JOURNALED_CAS,
    strategy_plan,
  )


def test_trusted_plan_approval_review_exposes_exact_workbook_change_and_undo() -> None:
  change_set = _planned_workbook_change_set()
  trusted_plan = dispatcher_module.TrustedToolPlan.create(
    identity_source="change_set",
    identity=change_set,
    prepared=SimpleNamespace(change_set=change_set),
  )

  review = trusted_plan.approval_review()

  assert review["schema_version"] == "planned-change-review.v1"
  assert review["change_set_id"] == change_set.change_set_id
  assert review["target"] == {
    "ticker": "PCTY",
    "research_file_id": 7,
    "workspace": "workspace/PCTY",
    "scope": "workspace",
  }
  assert review["commit_strategy"] == "JOURNALED_CAS"
  assert review["precommit_snapshot_store_ids"] == ["workbook"]
  assert review["workbook"]["will_write"] is True
  assert review["workbook"]["operation_count"] == 1
  write = review["workbook"]["writes"][0]
  assert write["precommit_snapshot_present"] is True
  assert write["execution"] == {
    "operations": [
      {
        "item_id": "tpl.a.unit_economics.gross_margin",
        "type": "set_value",
        "values": {"2027": 0.42, "2028": 0.43},
      }
    ],
    "force_overwrite": True,
  }
  assert write["expected_readback_digest"]
  assert review["undo"] == {
    "scope": "workbook_and_projection_state",
    "status": "durable_token_after_commit",
    "tool_name": "fms_undo_model_writer_commit",
    "issued_when": "authorized_commit_succeeds",
    "covers_store_ids": ["workbook"],
  }


def test_trusted_plan_approval_review_exposes_forecast_assumptions_undo() -> None:
  change_set = _planned_workbook_change_set("persist_forecast_assumptions")
  trusted_plan = dispatcher_module.TrustedToolPlan.create(
    identity_source="change_set",
    identity=change_set,
    prepared=SimpleNamespace(change_set=change_set),
  )

  assert trusted_plan.approval_review()["undo"] == {
    "scope": "workbook_and_projection_state",
    "status": "durable_token_after_commit",
    "tool_name": "fms_undo_model_writer_commit",
    "issued_when": "authorized_commit_succeeds",
    "covers_store_ids": ["projection_override", "workbook"],
  }


def _planned_handler(
  *,
  events: list[str],
  context_sink: dict[str, Any] | None = None,
) -> tuple[Any, ChangeSet, Any]:
  change_set = _planned_change_set()
  prepared = SimpleNamespace(change_set=change_set)

  async def legacy_handler(_tool_input: dict[str, Any], **_kwargs: Any):
    raise AssertionError("planned tools must not execute through the legacy handler")

  async def plan_change(
    tool_input: dict[str, Any],
    *,
    call_index: int,
    tool_ctx: Any,
  ) -> tuple[ChangeSet, Any]:
    assert tool_input == {"x": 1}
    assert call_index == 3
    assert tool_ctx.trusted_plan is None
    events.append("plan")
    if context_sink is not None:
      context_sink["ctx"] = tool_ctx
    return change_set, prepared

  async def execute_prepared_change(
    candidate: Any,
    *,
    authorized_identity: Any,
    approval_id: str,
    approval_chain_id: str,
    approval_request: Any,
    call_index: int,
    tool_ctx: Any,
  ):
    events.append("execute")
    assert candidate is prepared
    assert authorized_identity is change_set
    assert tool_ctx.trusted_plan.prepared is prepared
    assert tool_ctx.trusted_plan.identity is change_set
    assert approval_id == tool_ctx.approval_id
    assert approval_chain_id == tool_ctx.approval_chain_id
    assert approval_request.approval_id == approval_id
    assert approval_request.approval_chain_id == approval_chain_id
    assert approval_request.change_set_id == change_set.change_set_id
    assert call_index == 3
    return {"ok": True, "approval_id": approval_id}, None

  legacy_handler.PLANNING_IDENTITY = "change_set"
  legacy_handler.plan_change = plan_change
  legacy_handler.execute_prepared_change = execute_prepared_change
  return legacy_handler, change_set, prepared


def _planned_session() -> SimpleNamespace:
  return SimpleNamespace(
    session_id="sess-1",
    user_id="alice",
    channel="web",
    role="owner",
    pending_tools={},
    approval_queues={},
  )


def test_requires_approval_includes_durable_planned_write_wait() -> None:
  handler, _change_set, _prepared = _planned_handler(events=[])
  dispatcher = ToolDispatcher(
    mcp_client=_NullMcp(),
    local_tool_handlers={"planned_tool": handler},
    needs_approval=lambda *_args: False,
    approved_tool_types=set(),
    session=_planned_session(),
    store=object(),
    policy=object(),
  )

  assert dispatcher.requires_approval("planned_tool", {"x": 1}) is True


_DEFAULT_BUSINESS_MODEL_RESTORE = object()


def _business_model_planned_handler(
  *,
  receipt_status: str,
  restore_payload: object = _DEFAULT_BUSINESS_MODEL_RESTORE,
) -> tuple[Any, ChangeSet, Any]:
  change_set = _planned_change_set()
  prepared_bytes = b'{"schema_version":"prepared-business-model-accept.v1"}'
  prepared_accept = SimpleNamespace(
    caller_kind="fms_persist",
    user_scope="alice",
    idempotency_locator="skill-run-phase6",
    intent_digest="b" * 64,
    to_canonical_bytes=lambda: prepared_bytes,
  )
  outer = SimpleNamespace(
    change_set=change_set,
    completion=SimpleNamespace(
      finalizer=SimpleNamespace(prepared_accept=prepared_accept),
    ),
  )
  gateway_prepared = SimpleNamespace(change_set=change_set, prepared=outer)

  async def legacy_handler(_tool_input: dict[str, Any], **_kwargs: Any):
    raise AssertionError("planned tools must not execute through the legacy handler")

  async def plan_change(
    _tool_input: dict[str, Any],
    *,
    call_index: int,
    tool_ctx: Any,
  ):
    _ = call_index, tool_ctx
    return change_set, gateway_prepared

  async def execute_prepared_change(
    candidate: Any,
    **_kwargs: Any,
  ):
    assert candidate is gateway_prepared
    receipt = {
      "status": receipt_status,
      "child_refs": (
        [{"kind": "accept_checkpoint_id", "value": "c" * 64}]
        if receipt_status != "FAILED_PRECOMMIT"
        else []
      ),
    }
    if receipt_status == "FAILED_PRECOMMIT":
      restoration = (
        {
          "file": {
            "restored": True,
            "base_digest": "a" * 64,
            "target": "/tmp/business-model.md",
          }
        }
        if restore_payload is _DEFAULT_BUSINESS_MODEL_RESTORE
        else restore_payload
      )
      return {
        "status": "error",
        "receipt": receipt,
        "error": {
          "data": ({"restore": restoration} if restoration is not None else {})
        },
      }, None
    return {"status": "staged", "receipt": receipt}, None

  legacy_handler.PLANNING_IDENTITY = "change_set"
  legacy_handler.plan_change = plan_change
  legacy_handler.execute_prepared_change = execute_prepared_change
  return legacy_handler, change_set, gateway_prepared


@pytest.mark.parametrize(
  ("receipt_status", "expected_lifecycle"),
  [
    ("COMMITTED", PreparedBusinessModelLifecycle.CONSUMED),
    ("PARTIAL", PreparedBusinessModelLifecycle.CONSUMED),
    ("FAILED_PRECOMMIT", PreparedBusinessModelLifecycle.SUPERSEDED_PRECOMMIT),
  ],
)
def test_fms_business_model_exact_write_persists_attempt_lifecycle(
  tmp_path: Path,
  receipt_status: str,
  expected_lifecycle: PreparedBusinessModelLifecycle,
) -> None:
  handler, _change_set, _prepared = _business_model_planned_handler(
    receipt_status=receipt_status,
  )
  policy_calls = 0

  class Policy:
    policy_id = "phase6-test"
    policy_version = "1"

    async def decide(self, **_kwargs: Any):
      nonlocal policy_calls
      policy_calls += 1
      return PolicyApprovalDecision(outcome="auto_approve", reason="approved")

    async def on_resolve(self, **_kwargs: Any):
      return None

  store = SQLiteApprovalStore(tmp_path / "approvals.sqlite3")
  dispatcher = ToolDispatcher(
    mcp_client=_NullMcp(),
    local_tool_handlers={"fms_persist_business_model": handler},
    needs_approval=lambda *_args: True,
    approved_tool_types=set(),
    session=_planned_session(),
    store=store,
    policy=Policy(),
    run_context=RunContext(
      user_id="alice",
      request_id="skill-run-phase6",
      session_id="sess-1",
      profile="chat",
      channel="web",
      decider_role="owner",
    ),
  )

  _result, error = asyncio.run(
    dispatcher.dispatch(
      "call-phase6",
      "fms_persist_business_model",
      {"judgment": {"ticker": "TEST"}},
      skill_run_id="skill-run-phase6",
    )
  )

  assert error is None
  stored = asyncio.run(
    store.get_prepared_business_model_change(
      caller_kind="fms_persist",
      user_scope="alice",
      idempotency_locator="skill-run-phase6",
    )
  )
  assert stored is not None
  assert policy_calls == 1
  assert stored.lifecycle is expected_lifecycle
  assert stored.execution_receipt is not None
  if expected_lifecycle is PreparedBusinessModelLifecycle.CONSUMED:
    assert stored.checkpoint_id == "c" * 64
    replay_result, replay_error = asyncio.run(
      dispatcher.dispatch(
        "call-phase6-replay",
        "fms_persist_business_model",
        {"judgment": {"ticker": "TEST"}},
        skill_run_id="skill-run-phase6",
      )
    )
    assert replay_error is None
    assert replay_result["status"] == "staged"
    assert policy_calls == 1
  else:
    assert stored.restoration_digest is not None
    retry_result, retry_error = asyncio.run(
      dispatcher.dispatch(
        "call-phase6-retry",
        "fms_persist_business_model",
        {"judgment": {"ticker": "TEST"}},
        skill_run_id="skill-run-phase6",
      )
    )
    assert retry_result is None
    assert retry_error["code"] == "planned_write_replan_and_reauthorize_required"
    assert policy_calls == 1


@pytest.mark.parametrize(
  "restore_payload",
  [
    None,
    {},
    {"restored": False},
    {"file": {}},
    {"file": {"restored": False}},
    {"file": {"restored": True, "target": "/tmp/business-model.md"}},
    {
      "file": {
        "restored": True,
        "target": "/tmp/business-model.md",
        "base_digest": "not-a-digest",
      }
    },
  ],
)
def test_fms_business_model_missing_restoration_proof_does_not_supersede(
  tmp_path: Path,
  restore_payload: object,
) -> None:
  handler, _change_set, _prepared = _business_model_planned_handler(
    receipt_status="FAILED_PRECOMMIT",
    restore_payload=restore_payload,
  )

  class Policy:
    policy_id = "phase6-test"
    policy_version = "1"

    async def decide(self, **_kwargs: Any):
      return PolicyApprovalDecision(outcome="auto_approve", reason="approved")

    async def on_resolve(self, **_kwargs: Any):
      return None

  store = SQLiteApprovalStore(tmp_path / "approvals.sqlite3")
  dispatcher = ToolDispatcher(
    mcp_client=_NullMcp(),
    local_tool_handlers={"fms_persist_business_model": handler},
    needs_approval=lambda *_args: True,
    approved_tool_types=set(),
    session=_planned_session(),
    store=store,
    policy=Policy(),
    run_context=RunContext(
      user_id="alice",
      request_id="skill-run-phase6",
      session_id="sess-1",
      profile="chat",
      channel="web",
      decider_role="owner",
    ),
  )

  result, error = asyncio.run(
    dispatcher.dispatch(
      "call-phase6-missing-restore",
      "fms_persist_business_model",
      {"judgment": {"ticker": "TEST"}},
      skill_run_id="skill-run-phase6",
    )
  )

  assert result is None
  assert error["code"] == "planned_write_recovery_evidence_missing"
  stored = asyncio.run(
    store.get_prepared_business_model_change(
      caller_kind="fms_persist",
      user_scope="alice",
      idempotency_locator="skill-run-phase6",
    )
  )
  assert stored is not None
  assert stored.lifecycle is PreparedBusinessModelLifecycle.AUTHORIZED


@pytest.mark.parametrize(
  (
    "resolved_state",
    "prepared_ttl_elapsed",
    "concurrent_sweep",
    "expected_lifecycle",
    "expect_execution",
  ),
  [
    ("approved", False, False, PreparedBusinessModelLifecycle.CONSUMED, True),
    ("approved", False, True, PreparedBusinessModelLifecycle.CONSUMED, True),
    ("auto_approved", False, False, PreparedBusinessModelLifecycle.CONSUMED, True),
    ("denied", False, False, PreparedBusinessModelLifecycle.DENIED, False),
    ("auto_denied", False, False, PreparedBusinessModelLifecycle.DENIED, False),
    ("cancelled", False, False, PreparedBusinessModelLifecycle.DENIED, False),
    ("expired", False, False, PreparedBusinessModelLifecycle.EXPIRED, False),
    ("approved", True, False, PreparedBusinessModelLifecycle.EXPIRED, False),
  ],
)
def test_fms_business_model_pending_record_reconciles_original_approval(
  tmp_path: Path,
  resolved_state: str,
  prepared_ttl_elapsed: bool,
  concurrent_sweep: bool,
  expected_lifecycle: PreparedBusinessModelLifecycle,
  expect_execution: bool,
) -> None:
  handler, _change_set, _prepared = _business_model_planned_handler(
    receipt_status="COMMITTED",
  )
  policy_calls = 0

  class Policy:
    policy_id = "phase6-test"
    policy_version = "1"

    async def decide(self, **_kwargs: Any):
      nonlocal policy_calls
      policy_calls += 1
      return PolicyApprovalDecision(
        outcome="route_external",
        reason="route exact plan",
        route_target="phase6-review",
        route_target_type="service",
        expiry_seconds=600,
      )

    async def on_resolve(self, **_kwargs: Any):
      return None

  store = SQLiteApprovalStore(tmp_path / "approvals.sqlite3")
  dispatcher = ToolDispatcher(
    mcp_client=_NullMcp(),
    local_tool_handlers={"fms_persist_business_model": handler},
    needs_approval=lambda *_args: True,
    approved_tool_types=set(),
    session=_planned_session(),
    store=store,
    policy=Policy(),
    run_context=RunContext(
      user_id="alice",
      request_id="skill-run-phase6",
      session_id="sess-1",
      profile="chat",
      channel="web",
      decider_role="owner",
    ),
  )

  first_result, first_error = asyncio.run(
    dispatcher.dispatch(
      "call-phase6-route",
      "fms_persist_business_model",
      {"judgment": {"ticker": "TEST"}},
      skill_run_id="skill-run-phase6",
    )
  )
  assert first_result is None
  assert first_error["code"] == "approval_timeout"
  pending = asyncio.run(
    store.get_prepared_business_model_change(
      caller_kind="fms_persist",
      user_scope="alice",
      idempotency_locator="skill-run-phase6",
    )
  )
  assert pending is not None
  assert pending.lifecycle is PreparedBusinessModelLifecycle.PENDING
  original_approval_id = pending.approval_id
  original = asyncio.run(store.get(original_approval_id))
  assert original is not None
  if prepared_ttl_elapsed:
    with store._connection() as conn:
      conn.execute(
        """
        UPDATE prepared_business_model_change
        SET expires_at = ?
        WHERE caller_kind = ? AND user_scope = ? AND idempotency_locator = ?
        """,
        (
          (datetime.now(UTC) - timedelta(seconds=1)).isoformat(),
          pending.caller_kind,
          pending.user_scope,
          pending.idempotency_locator,
        ),
      )
  asyncio.run(
    store.transition_state(
      original_approval_id,
      resolved_state,
      expected_state_version=original.state_version,
    )
  )

  async def retry_with_optional_sweep():
    retry = dispatcher.dispatch(
      "call-phase6-resume",
      "fms_persist_business_model",
      {"judgment": {"ticker": "TEST"}},
      skill_run_id="skill-run-phase6",
    )
    if not concurrent_sweep:
      return await retry
    independent_store = SQLiteApprovalStore(store.path)
    retry_result, _maintenance = await asyncio.gather(
      retry,
      independent_store.maintain_pending(),
    )
    return retry_result

  result, error = asyncio.run(retry_with_optional_sweep())

  if expect_execution:
    assert error is None
    assert result["status"] == "staged"
  else:
    assert result is None
    assert error["code"] == "planned_write_authorization_state_invalid"
  reconciled = asyncio.run(
    store.get_prepared_business_model_change(
      caller_kind="fms_persist",
      user_scope="alice",
      idempotency_locator="skill-run-phase6",
    )
  )
  assert reconciled is not None
  assert reconciled.lifecycle is expected_lifecycle
  assert reconciled.approval_id == original_approval_id
  assert policy_calls == 1


def test_batch_admission_cancel_between_pending_commit_and_publish_cleans_all_residue(
  tmp_path: Path,
) -> None:
  async def run_case() -> None:
    class Policy:
      policy_id = "batch-admission-test"
      policy_version = "1"
      policy_bundle_hash = "batch-admission-bundle"

      def __init__(self) -> None:
        self.request = None

      async def decide(self, *, payload, request, run_context):
        _ = payload, run_context
        self.request = request
        return PolicyApprovalDecision(
          outcome="request_user_approval",
          reason="user review required",
          route_target_type="pending_tools",
          expiry_seconds=600,
        )

      async def on_resolve(self, *, request):
        _ = request

    store = SQLiteApprovalStore(tmp_path / "admission-abort.sqlite3")
    policy = Policy()
    registry = BatchApprovalProjectionRegistry()
    session = SimpleNamespace(
      session_id="batch-stage-admission-abort",
      user_id="alice",
      channel="tui",
      role="owner",
      batch_stage_run_seq=3,
      pending_tools={},
      approval_queues={},
    )
    scope = BatchApprovalScope(
      batch_id=41,
      owner_user_id="alice",
      channel="tui",
      store=store,
      policy=policy,
      registry=registry,
    )
    scope.register_session(session)
    session.batch_approval_scope = scope
    dispatcher = ToolDispatcher(
      mcp_client=_NullMcp(),
      local_tool_handlers={},
      event_log=EventLog(),
      session=session,
      store=store,
      policy=policy,
      run_context=RunContext(
        user_id="alice",
        request_id="batch_41",
        run_id="batch_41",
        session_id=session.session_id,
        profile="analyst",
        channel="tui",
        decider_role="owner",
      ),
    )
    original_enqueue = store.enqueue_pending_approval_notification
    pending_committed = asyncio.Event()

    async def pause_after_notification_commit(request):
      result = await original_enqueue(request)
      pending_committed.set()
      await asyncio.Event().wait()
      return result

    store.enqueue_pending_approval_notification = pause_after_notification_commit  # type: ignore[method-assign]
    lifecycle_task = asyncio.create_task(
      dispatcher._run_approval_lifecycle(
        tool_call_id="tool-admission-abort",
        tool_name="memory_write",
        tool_input={"file": "notes/admission.md"},
        qualifier="",
        reason="batch approval test",
        allow_persistent=False,
      )
    )
    await pending_committed.wait()
    lifecycle_task.cancel()
    with pytest.raises(asyncio.CancelledError):
      await lifecycle_task

    assert policy.request is not None
    stored = await store.get(policy.request.approval_id)
    assert stored is not None
    assert stored.state == "denied"
    assert session.pending_tools == {}
    assert session.approval_queues == {}
    assert await store.list_approval_notification_outbox(policy.request.approval_id) == []
    assert registry.projections_for_batch(owner_user_id="alice", batch_id=41) == []
    assert registry._admission_gates[("alice", 41)].active == 0

  asyncio.run(run_case())


@pytest.mark.parametrize(
  ("missing_capability", "expected_error"),
  [
    ("abort_unpublished_approval", "cannot abort unpublished approvals"),
    (
      "fence_persistent_grants_for_cancellation",
      "cannot quarantine persistent approval grants",
    ),
    (
      "revoke_persistent_grants_for_approval",
      "cannot quarantine persistent approval grants",
    ),
  ],
)
def test_batch_admission_requires_cleanup_store_contract(
  missing_capability: str,
  expected_error: str,
) -> None:
  async def run_case() -> None:
    class StoreWithCleanup:
      def __init__(self) -> None:
        self.created = False

      async def create(self, request):
        _ = request
        self.created = True

      async def abort_unpublished_approval(self, *args, **kwargs):
        _ = args, kwargs

      async def fence_persistent_grants_for_cancellation(self, *args, **kwargs):
        _ = args, kwargs

      async def revoke_persistent_grants_for_approval(self, *args, **kwargs):
        _ = args, kwargs

    class Policy:
      policy_id = "missing-abort-test"
      policy_version = "1"
      policy_bundle_hash = "missing-abort-bundle"

    store = StoreWithCleanup()
    setattr(store, missing_capability, None)
    policy = Policy()
    registry = BatchApprovalProjectionRegistry()
    session = SimpleNamespace(
      session_id="batch-stage-missing-abort",
      user_id="alice",
      channel="tui",
      role="owner",
      batch_stage_run_seq=3,
      pending_tools={},
      approval_queues={},
    )
    scope = BatchApprovalScope(
      batch_id=42,
      owner_user_id="alice",
      channel="tui",
      store=store,
      policy=policy,
      registry=registry,
    )
    scope.register_session(session)
    session.batch_approval_scope = scope
    dispatcher = ToolDispatcher(
      mcp_client=_NullMcp(),
      local_tool_handlers={},
      session=session,
      store=store,
      policy=policy,
      run_context=RunContext(
        user_id="alice",
        request_id="batch_42",
        run_id="batch_42",
        session_id=session.session_id,
        profile="analyst",
        channel="tui",
        decider_role="owner",
      ),
    )

    with pytest.raises(
      RuntimeError,
      match=expected_error,
    ):
      await dispatcher._run_approval_lifecycle(
        tool_call_id="tool-missing-abort",
        tool_name="memory_write",
        tool_input={},
        qualifier="",
        reason="batch approval test",
        allow_persistent=False,
      )
    assert not store.created
    assert registry._admission_gates[("alice", 42)].active == 0

  asyncio.run(run_case())


@pytest.mark.parametrize("with_event_log", [False, True])
def test_planned_dispatch_persists_identity_and_executes_exact_objects(
  tmp_path: Path,
  with_event_log: bool,
) -> None:
  events: list[str] = []
  resolved: list[Any] = []
  handler, change_set, prepared = _planned_handler(events=events)

  class Store(SQLiteApprovalStore):
    async def create(self, request):
      events.append("row")
      return await super().create(request)

  class Policy:
    policy_id = "planned-test"
    policy_version = "1"
    policy_bundle_hash = "bundle"

    async def decide(self, *, payload, request, run_context):
      _ = payload, run_context
      events.append("policy")
      assert request.identity_source == "change_set"
      assert request.change_set_id == change_set.change_set_id
      review = request.tool_args_redacted["planned_change"]
      assert review["change_set_id"] == change_set.change_set_id
      assert review["commit_strategy"] == "ARTIFACT_ONLY"
      assert review["workbook"] == {
        "will_write": False,
        "operation_count": 0,
        "writes": [],
      }
      assert review["undo"] == {
        "scope": "workbook_and_projection_state",
        "status": "not_required",
        "reason": "plan_has_no_workbook_or_projection_state_write",
      }
      return PolicyApprovalDecision(outcome="auto_approve", reason="exact plan approved")

    async def on_resolve(self, *, request):
      resolved.append(request)

  store = Store(tmp_path / "approvals.sqlite3")
  dispatcher = ToolDispatcher(
    mcp_client=_NullMcp(),
    local_tool_handlers={"planned": handler},
    needs_approval=lambda *_args: True,
    approved_tool_types=set(),
    event_log=EventLog() if with_event_log else None,
    session=_planned_session(),
    store=store,
    policy=Policy(),
    run_context=RunContext(
      user_id="alice",
      request_id="req-1",
      session_id="sess-1",
      profile="chat",
      channel="web",
      decider_role="owner",
    ),
  )

  result, error = asyncio.run(
    dispatcher.dispatch("call-1", "planned", {"x": 1}, call_index=3)
  )

  assert error is None
  assert result["ok"] is True
  assert events == ["plan", "row", "policy", "execute"]
  assert len(resolved) == 1
  request = resolved[0]
  assert request.state == "auto_approved"
  assert request.change_set_id == change_set.change_set_id
  assert request.change_hash == change_set.change_hash
  assert request.base_vector_hash
  assert request.reviewed_change_binding_digest is None
  assert request.execution_semantics_digest is None
  assert asyncio.run(store.get(request.approval_id)) == request
  assert prepared.change_set is change_set


def test_planned_dispatch_session_cache_still_creates_bound_row(tmp_path: Path) -> None:
  events: list[str] = []
  resolved: list[Any] = []
  handler, change_set, _prepared = _planned_handler(events=events)

  class Policy:
    policy_id = "planned-test"
    policy_version = "1"

    async def decide(self, **_kwargs: Any):
      raise AssertionError("session cache must resolve without another policy decision")

    async def on_resolve(self, *, request):
      resolved.append(request)

  store = SQLiteApprovalStore(tmp_path / "approvals.sqlite3")
  dispatcher = ToolDispatcher(
    mcp_client=_NullMcp(),
    local_tool_handlers={"planned": handler},
    needs_approval=lambda *_args: True,
    approved_tool_types={"planned"},
    session=_planned_session(),
    store=store,
    policy=Policy(),
  )

  result, error = asyncio.run(
    dispatcher.dispatch("call-cache", "planned", {"x": 1}, call_index=3)
  )

  assert error is None
  assert result["ok"] is True
  assert events == ["plan", "execute"]
  assert len(resolved) == 1
  request = resolved[0]
  assert request.state == "auto_approved"
  assert request.change_set_id == change_set.change_set_id
  assert asyncio.run(store.get(request.approval_id)) == request


def test_planned_dispatch_persistent_grant_still_creates_bound_row(
  tmp_path: Path,
) -> None:
  events: list[str] = []
  handler, change_set, _prepared = _planned_handler(events=events)
  store = SQLiteApprovalStore(tmp_path / "approvals.sqlite3")
  policy = SingleUserApprovalPolicy(store=store)
  dispatcher = ToolDispatcher(
    mcp_client=_NullMcp(),
    local_tool_handlers={"planned": handler},
    needs_approval=lambda *_args: True,
    session=_planned_session(),
    store=store,
    policy=policy,
  )
  scope_hint = f"{dispatcher._resolve_tool_class('planned')}:planned"
  prior_approval = replace(
    build_approval_request(
      tool_call_id="prior-call",
      tool_name="planned",
      tool_class="state_write",
      tool_args_redacted={},
      args_hash="prior-args",
      run_context=RunContext(user_id="alice", request_id="prior-request"),
    ),
    approval_id="prior-approval",
    approval_chain_id="prior-approval",
    state="approved",
    decision="approved",
    decided_at=utc_now(),
    decider_id="alice",
    decider_role="owner",
    persistent_grant_scope=scope_hint,
  )
  asyncio.run(store.create(prior_approval))
  asyncio.run(
    store.create_persistent_grant(
      PersistentGrant(
        grant_id="grant-planned",
        user_id="alice",
        tool_name="planned",
        scope_hint=scope_hint,
        args_predicate=None,
        granted_at=utc_now(),
        expires_at=None,
        revoked_at=None,
        granted_via_approval_id="prior-approval",
        policy_id=policy.policy_id,
      )
    )
  )

  result, error = asyncio.run(
    dispatcher.dispatch("call-grant", "planned", {"x": 1}, call_index=3)
  )

  assert error is None
  assert result["ok"] is True
  assert events == ["plan", "execute"]
  request = asyncio.run(store.get(result["approval_id"]))
  assert request is not None
  assert request.approval_id != prior_approval.approval_id
  assert request.state == "auto_approved"
  assert request.authorization_mode == "PERSISTENT_GRANT"
  assert request.grant_reference == "grant-planned"
  assert request.identity_source == "change_set"
  assert request.change_set_id == change_set.change_set_id
  assert request.change_hash == change_set.change_hash
  assert request.base_vector_hash
  assert request.persistent_grant_scope == scope_hint
  assert request.decision_reason == "Persistent approval grant matched"


def test_planned_dispatch_headless_denial_still_creates_bound_row(tmp_path: Path) -> None:
  events: list[str] = []
  resolved: list[Any] = []
  handler, change_set, _prepared = _planned_handler(events=events)

  class Policy:
    async def decide(self, **_kwargs: Any):
      raise AssertionError("static headless denial must resolve before policy evaluation")

    async def on_resolve(self, *, request):
      resolved.append(request)

  store = SQLiteApprovalStore(tmp_path / "approvals.sqlite3")
  dispatcher = ToolDispatcher(
    mcp_client=_NullMcp(),
    local_tool_handlers={"planned": handler},
    needs_approval=lambda *_args: True,
    should_avoid_permission_prompts=True,
    session=_planned_session(),
    store=store,
    policy=Policy(),
  )

  result, error = asyncio.run(
    dispatcher.dispatch("call-headless", "planned", {"x": 1}, call_index=3)
  )

  assert result is None
  assert error["code"] == "headless_auto_deny"
  assert events == ["plan"]
  assert len(resolved) == 1
  request = resolved[0]
  assert request.state == "auto_denied"
  assert request.change_set_id == change_set.change_set_id
  assert asyncio.run(store.get(request.approval_id)) == request


def test_planned_dispatch_headless_autonomous_allow_creates_bound_row(
  tmp_path: Path,
) -> None:
  events: list[str] = []
  resolved: list[Any] = []
  handler, change_set, _prepared = _planned_handler(events=events)

  class Policy:
    async def decide(self, **_kwargs: Any):
      raise AssertionError("autonomous allow must resolve before policy evaluation")

    async def on_resolve(self, *, request):
      resolved.append(request)

  store = SQLiteApprovalStore(tmp_path / "approvals.sqlite3")
  dispatcher = ToolDispatcher(
    mcp_client=_NullMcp(),
    local_tool_handlers={"planned": handler},
    needs_approval=lambda *_args: False,
    should_avoid_permission_prompts=True,
    session=_planned_session(),
    store=store,
    policy=Policy(),
  )

  result, error = asyncio.run(
    dispatcher.dispatch("call-autonomous", "planned", {"x": 1}, call_index=3)
  )

  assert error is None
  assert result["ok"] is True
  assert events == ["plan", "execute"]
  assert len(resolved) == 1
  request = resolved[0]
  assert request.state == "auto_approved"
  assert request.change_set_id == change_set.change_set_id
  assert request.decision_reason == (
    "Autonomous tool policy authorized the exact planned identity"
  )
  assert asyncio.run(store.get(request.approval_id)) == request


def test_planned_dispatch_timeout_preserves_bound_row_and_skips_executor(
  tmp_path: Path,
) -> None:
  events: list[str] = []
  planned_requests: list[Any] = []
  handler, change_set, _prepared = _planned_handler(events=events)

  class Policy:
    async def decide(self, *, payload, request, run_context):
      _ = payload, run_context
      planned_requests.append(request)
      return PolicyApprovalDecision(
        outcome="request_user_approval",
        reason="approval required",
        route_target_type="pending_tools",
        expiry_seconds=0.001,
        allow_persistent_grant=True,
      )

    async def on_resolve(self, *, request):
      raise AssertionError(f"timed-out request must not resolve: {request}")

  session = _planned_session()
  store = SQLiteApprovalStore(tmp_path / "approvals.sqlite3")
  dispatcher = ToolDispatcher(
    mcp_client=_NullMcp(),
    local_tool_handlers={"planned": handler},
    needs_approval=lambda *_args: True,
    session=session,
    store=store,
    policy=Policy(),
  )

  result, error = asyncio.run(
    dispatcher.dispatch("call-timeout", "planned", {"x": 1}, call_index=3)
  )

  assert result is None
  assert error["code"] == "approval_timeout"
  assert events == ["plan"]
  assert len(planned_requests) == 1
  stored = asyncio.run(store.get(planned_requests[0].approval_id))
  assert stored is not None
  assert stored.state == "expired"
  assert stored.identity_source == "change_set"
  assert stored.change_set_id == change_set.change_set_id
  assert session.pending_tools == {}
  assert session.approval_queues == {}


@pytest.mark.parametrize(
  ("callback", "expected_code"),
  [
    (None, "planned_write_authorization_unavailable"),
    (lambda _request: None, "planned_write_callback_transport_unsupported"),
  ],
)
def test_planned_dispatch_fails_closed_without_durable_lifecycle(
  callback: Any,
  expected_code: str,
) -> None:
  events: list[str] = []
  handler, _change_set, _prepared = _planned_handler(events=events)
  dispatcher = ToolDispatcher(
    mcp_client=_NullMcp(),
    local_tool_handlers={"planned": handler},
    needs_approval=lambda *_args: True,
    request_approval=callback,
  )

  result, error = asyncio.run(
    dispatcher.dispatch("call-1", "planned", {"x": 1}, call_index=3)
  )

  assert result is None
  assert error["code"] == expected_code
  assert events == ["plan"]


def test_raw_patch_mcp_dry_run_bypasses_write_planning_and_authorization() -> None:
  calls: list[dict[str, Any]] = []

  class Mcp(_NullMcp):
    def is_mcp_tool(self, name: str) -> bool:
      return name == "apply_patch_ops"

    def get_server_for_tool(self, name: str) -> str | None:
      return "portfolio-writes-mcp" if name == "apply_patch_ops" else None

    async def call_tool(self, _name: str, tool_input: dict, **_kwargs: Any):
      calls.append(dict(tool_input))
      return {"dry_run": True}, None

  dispatcher = ToolDispatcher(
    mcp_client=Mcp(),
    needs_approval=lambda *_args: True,
  )
  tool_input = {"research_file_id": 42, "ops": [], "dry_run": True}

  result, error = asyncio.run(
    dispatcher.dispatch("dry-run-call", "apply_patch_ops", tool_input)
  )

  assert error is None
  assert result == {"dry_run": True}
  assert calls == [tool_input]
  assert "authorization_ref" not in calls[0]


def _missing_catalog_module(name: str) -> ModuleNotFoundError:
  return ModuleNotFoundError(f"No module named {name!r}", name=name)


def _catalog_module(*tool_names: str) -> SimpleNamespace:
  return SimpleNamespace(
    ACTION_CATALOG=tuple(
      SimpleNamespace(
        local_tool_name=tool_name,
        planning_identity="change_set",
      )
      for tool_name in tool_names
    )
  )


def test_catalog_planning_identity_prefers_fms_layout(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  calls: list[str] = []

  def fake_import_module(name: str) -> Any:
    calls.append(name)
    if name != "fms.action_catalog":
      raise AssertionError("the fallback catalog must not be merged")
    return _catalog_module("planned")

  monkeypatch.setattr(dispatcher_module, "import_module", fake_import_module)

  assert ToolDispatcher._catalog_planning_identity("planned") == "change_set"
  assert calls == ["fms.action_catalog"]


@pytest.mark.parametrize("missing_name", ["fms", "fms.action_catalog"])
def test_catalog_planning_identity_falls_back_to_api_layout_only_for_candidate_absence(
  monkeypatch: pytest.MonkeyPatch,
  missing_name: str,
) -> None:
  calls: list[str] = []

  def fake_import_module(name: str) -> Any:
    calls.append(name)
    if name == "fms.action_catalog":
      raise _missing_catalog_module(missing_name)
    return _catalog_module("planned")

  monkeypatch.setattr(dispatcher_module, "import_module", fake_import_module)

  assert ToolDispatcher._catalog_planning_identity("planned") == "change_set"
  assert calls == ["fms.action_catalog", "api.fms.action_catalog"]


@pytest.mark.parametrize(
  "api_missing_name",
  ["api", "api.fms", "api.fms.action_catalog"],
)
def test_catalog_planning_identity_fails_closed_when_both_layouts_are_absent(
  monkeypatch: pytest.MonkeyPatch,
  api_missing_name: str,
) -> None:
  calls: list[str] = []

  def fake_import_module(name: str) -> Any:
    calls.append(name)
    missing_name = "fms" if name == "fms.action_catalog" else api_missing_name
    raise _missing_catalog_module(missing_name)

  monkeypatch.setattr(dispatcher_module, "import_module", fake_import_module)

  with pytest.raises(
    dispatcher_module.TrustedToolPlanError,
    match="trusted FMS action catalog is unavailable",
  ):
    ToolDispatcher._catalog_planning_identity("planned")
  assert calls == ["fms.action_catalog", "api.fms.action_catalog"]


def test_catalog_planning_identity_rethrows_nested_dependency_failure(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  error = _missing_catalog_module("catalog_dependency")
  calls: list[str] = []

  def fake_import_module(name: str) -> Any:
    calls.append(name)
    raise error

  monkeypatch.setattr(dispatcher_module, "import_module", fake_import_module)

  with pytest.raises(ModuleNotFoundError) as exc_info:
    ToolDispatcher._catalog_planning_identity("planned")
  assert exc_info.value is error
  assert calls == ["fms.action_catalog"]


def test_catalog_planning_identity_rethrows_fallback_nested_dependency_failure(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  error = _missing_catalog_module("catalog_dependency")
  calls: list[str] = []

  def fake_import_module(name: str) -> Any:
    calls.append(name)
    if name == "fms.action_catalog":
      raise _missing_catalog_module("fms")
    raise error

  monkeypatch.setattr(dispatcher_module, "import_module", fake_import_module)

  with pytest.raises(ModuleNotFoundError) as exc_info:
    ToolDispatcher._catalog_planning_identity("planned")
  assert exc_info.value is error
  assert calls == ["fms.action_catalog", "api.fms.action_catalog"]


def test_catalog_planning_identity_rethrows_other_import_failure(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  error = RuntimeError("catalog import failed")

  def fake_import_module(_name: str) -> Any:
    raise error

  monkeypatch.setattr(dispatcher_module, "import_module", fake_import_module)

  with pytest.raises(RuntimeError) as exc_info:
    ToolDispatcher._catalog_planning_identity("planned")
  assert exc_info.value is error


def test_loaded_catalog_true_miss_does_not_consult_fallback(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  calls: list[str] = []

  def fake_import_module(name: str) -> Any:
    calls.append(name)
    if name != "fms.action_catalog":
      raise AssertionError("a loaded catalog miss must be authoritative")
    return _catalog_module()

  monkeypatch.setattr(dispatcher_module, "import_module", fake_import_module)

  assert ToolDispatcher._catalog_planning_identity("ordinary") is None
  assert calls == ["fms.action_catalog"]


def test_loaded_catalog_duplicate_row_remains_a_typed_contract_error(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  monkeypatch.setattr(
    dispatcher_module,
    "import_module",
    lambda _name: _catalog_module("planned", "planned"),
  )

  with pytest.raises(
    dispatcher_module.TrustedToolPlanError,
    match="duplicate local tool",
  ):
    ToolDispatcher._catalog_planning_identity("planned")


def test_missing_catalog_blocks_even_generic_local_handler(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  calls: list[str] = []

  def fake_import_module(name: str) -> Any:
    missing_name = "fms" if name == "fms.action_catalog" else "api"
    raise _missing_catalog_module(missing_name)

  async def handler(*_args: Any, **_kwargs: Any):
    calls.append("handler")
    return {"status": "unexpected"}, None

  monkeypatch.setattr(dispatcher_module, "import_module", fake_import_module)
  dispatcher = ToolDispatcher(
    mcp_client=_NullMcp(),
    local_tool_handlers={"ordinary": handler},
  )

  result, error = asyncio.run(dispatcher.dispatch("call-ordinary", "ordinary", {}))

  assert result is None
  assert error["code"] == "planned_write_contract_invalid"
  assert calls == []


def test_loaded_catalog_true_miss_preserves_generic_local_handler(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  calls: list[str] = []

  async def handler(*_args: Any, **_kwargs: Any):
    calls.append("handler")
    return {"status": "ok"}, None

  monkeypatch.setattr(
    dispatcher_module,
    "import_module",
    lambda _name: _catalog_module(),
  )
  dispatcher = ToolDispatcher(
    mcp_client=_NullMcp(),
    local_tool_handlers={"ordinary": handler},
  )

  result, error = asyncio.run(dispatcher.dispatch("call-ordinary", "ordinary", {}))

  assert error is None
  assert result == {"status": "ok"}
  assert calls == ["handler"]


@pytest.mark.parametrize(
  "approval_path",
  ["session_cache", "headless", "custom_auto", "no_lifecycle"],
)
def test_promotion_saga_generic_dispatch_requires_owner_control_route_before_approval_paths(
  tmp_path: Path,
  approval_path: str,
) -> None:
  calls: list[str] = []

  async def handler(*_args: Any, **_kwargs: Any):
    calls.append("handler")
    return {"status": "unexpected"}, None

  class Store(SQLiteApprovalStore):
    async def create(self, request):
      calls.append("store")
      return await super().create(request)

  class Policy:
    policy_id = "promotion-test"
    policy_version = "1"
    policy_bundle_hash = "promotion-test-bundle"

    async def decide(self, **_kwargs: Any):
      calls.append("policy")
      return PolicyApprovalDecision(
        outcome="auto_approve",
        reason="custom policy attempted automatic promotion",
      )

    async def on_resolve(self, **_kwargs: Any):
      calls.append("resolve")

  dispatcher_kwargs: dict[str, Any] = {
    "mcp_client": _NullMcp(),
    "local_tool_handlers": {"merge_diligence_pr": handler},
    "needs_approval": lambda *_args: approval_path != "no_lifecycle",
  }
  if approval_path != "no_lifecycle":
    dispatcher_kwargs.update(
      session=_planned_session(),
      store=Store(tmp_path / "approvals.sqlite3"),
      policy=Policy(),
    )
  if approval_path == "session_cache":
    dispatcher_kwargs["approved_tool_types"] = {"merge_diligence_pr"}
  if approval_path == "headless":
    dispatcher_kwargs["should_avoid_permission_prompts"] = True

  dispatcher = ToolDispatcher(**dispatcher_kwargs)
  result, error = asyncio.run(
    dispatcher.dispatch(
      "call-promotion",
      "merge_diligence_pr",
      {"pr_id": "dpr-1", "confirm_merge": True},
    )
  )

  assert result is None
  assert error["code"] == "owner_control_route_required"
  assert "authenticated owner control-plane route" in error["message"].lower()
  assert calls == []


@pytest.mark.parametrize(
  "declared",
  [
    frozenset({"identity", "planner"}),
    frozenset({"identity", "executor"}),
    frozenset({"planner", "executor"}),
  ],
)
def test_planned_dispatch_partial_hook_triplet_fails_closed(
  declared: frozenset[str],
) -> None:
  calls: list[str] = []

  async def handler(*_args: Any, **_kwargs: Any):
    calls.append("legacy")
    return {}, None

  async def planner(*_args: Any, **_kwargs: Any):
    calls.append("plan")
    return None

  async def executor(*_args: Any, **_kwargs: Any):
    calls.append("execute")
    return {}, None

  if "identity" in declared:
    handler.PLANNING_IDENTITY = "change_set"
  if "planner" in declared:
    handler.plan_change = planner
  if "executor" in declared:
    handler.execute_prepared_change = executor

  dispatcher = ToolDispatcher(
    mcp_client=_NullMcp(),
    local_tool_handlers={"planned": handler},
  )

  result, error = asyncio.run(
    dispatcher.dispatch("call-partial", "planned", {})
  )

  assert result is None
  assert error["code"] == "planned_write_contract_invalid"
  assert calls == []


@pytest.mark.parametrize(
  "tool_name",
  ["apply_patch_proposal", "fms_persist_dcf_relative_valuation"],
)
def test_catalogued_exact_write_cannot_fall_back_to_legacy_handler(
  tool_name: str,
) -> None:
  calls: list[str] = []

  async def legacy_handler(*_args: Any, **_kwargs: Any):
    calls.append("legacy")
    return {"status": "unexpected"}, None

  dispatcher = ToolDispatcher(
    mcp_client=_NullMcp(),
    local_tool_handlers={tool_name: legacy_handler},
  )

  result, error = asyncio.run(
    dispatcher.dispatch("call-missing-exact-hooks", tool_name, {})
  )

  assert result is None
  assert error["code"] == "planned_write_contract_invalid"
  assert calls == []


def test_planned_dispatch_preserves_trusted_planning_rejection() -> None:
  calls: list[str] = []

  async def handler(_tool_input: dict[str, Any], **_kwargs: Any):
    raise AssertionError("planning rejection must not enter the legacy handler")

  async def plan_change(
    _tool_input: dict[str, Any],
    **_kwargs: Any,
  ) -> tuple[Any, Any]:
    calls.append("plan")
    raise PlannedWritePlanningRejected(
      error={
        "code": "invalid_model_writer_judgment",
        "message": "judgment requires a normalized scenario vector",
        "details": {"field": "judgment.scenarios"},
      }
    )

  async def execute_prepared_change(*_args: Any, **_kwargs: Any):
    calls.append("execute")
    return {}, None

  handler.PLANNING_IDENTITY = "change_set"
  handler.plan_change = plan_change
  handler.execute_prepared_change = execute_prepared_change
  dispatcher = ToolDispatcher(
    mcp_client=_NullMcp(),
    local_tool_handlers={"planned": handler},
    needs_approval=lambda *_args: True,
  )

  result, error = asyncio.run(
    dispatcher.dispatch("call-rejected", "planned", {"bad": True})
  )

  assert result is None
  assert error == {
    "code": "invalid_model_writer_judgment",
    "message": "judgment requires a normalized scenario vector",
    "details": {"field": "judgment.scenarios"},
  }
  assert calls == ["plan"]


def test_planned_dispatch_fails_closed_on_row_persistence_error(tmp_path: Path) -> None:
  events: list[str] = []
  handler, _change_set, _prepared = _planned_handler(events=events)

  class Store(SQLiteApprovalStore):
    async def create(self, request):
      _ = request
      raise RuntimeError("persistence unavailable")

  class Policy:
    async def decide(self, **_kwargs: Any):
      raise AssertionError("policy must not run after persistence failure")

  dispatcher = ToolDispatcher(
    mcp_client=_NullMcp(),
    local_tool_handlers={"planned": handler},
    needs_approval=lambda *_args: True,
    session=_planned_session(),
    store=Store(tmp_path / "approvals.sqlite3"),
    policy=Policy(),
  )

  result, error = asyncio.run(
    dispatcher.dispatch("call-1", "planned", {"x": 1}, call_index=3)
  )

  assert result is None
  assert error["code"] == "planned_write_authorization_persistence_failed"
  assert events == ["plan"]


def test_planned_dispatch_rejects_trusted_context_loss(tmp_path: Path) -> None:
  events: list[str] = []
  context_sink: dict[str, Any] = {}
  handler, _change_set, _prepared = _planned_handler(
    events=events,
    context_sink=context_sink,
  )

  class Policy:
    async def decide(self, **_kwargs: Any):
      context_sink["ctx"].trusted_plan = None
      return PolicyApprovalDecision(outcome="auto_approve", reason="approved")

    async def on_resolve(self, **_kwargs: Any):
      return None

  dispatcher = ToolDispatcher(
    mcp_client=_NullMcp(),
    local_tool_handlers={"planned": handler},
    needs_approval=lambda *_args: True,
    session=_planned_session(),
    store=SQLiteApprovalStore(tmp_path / "approvals.sqlite3"),
    policy=Policy(),
  )

  result, error = asyncio.run(
    dispatcher.dispatch("call-1", "planned", {"x": 1}, call_index=3)
  )

  assert result is None
  assert error["code"] == "planned_write_trusted_plan_lost"
  assert events == ["plan"]


def test_planned_dispatch_requires_reinvocation_for_policy_modified_args(
  tmp_path: Path,
) -> None:
  events: list[str] = []
  resolved: list[Any] = []
  handler, change_set, _prepared = _planned_handler(events=events)

  class Policy:
    async def decide(self, **_kwargs: Any):
      return PolicyApprovalDecision(
        outcome="auto_approve",
        reason="normalize input",
        modified_tool_args={"x": 2},
      )

    async def on_resolve(self, *, request):
      resolved.append(request)

  store = SQLiteApprovalStore(tmp_path / "approvals.sqlite3")
  dispatcher = ToolDispatcher(
    mcp_client=_NullMcp(),
    local_tool_handlers={"planned": handler},
    needs_approval=lambda *_args: True,
    session=_planned_session(),
    store=store,
    policy=Policy(),
  )

  result, error = asyncio.run(
    dispatcher.dispatch("call-1", "planned", {"x": 1}, call_index=3)
  )

  assert result is None
  assert error["code"] == "planned_write_reinvocation_required"
  assert events == ["plan"]
  assert len(resolved) == 1
  request = resolved[0]
  assert request.state == "auto_approved"
  assert request.change_set_id == change_set.change_set_id
  assert asyncio.run(store.get(request.approval_id)) == request


def test_local_handler_receives_context_when_event_logging_is_disabled() -> None:
  seen: dict[str, Any] = {}

  async def handler(_tool_input: dict[str, Any], *, tool_ctx: Any, **_kwargs: Any):
    seen["ctx"] = tool_ctx
    tool_ctx.emit({"type": "ignored"})
    return {"ok": True}, None

  dispatcher = ToolDispatcher(
    mcp_client=_NullMcp(),
    local_tool_handlers={"read": handler},
    event_log=None,
  )

  result, error = asyncio.run(dispatcher.dispatch("call-1", "read", {}))

  assert error is None
  assert result == {"ok": True}
  assert seen["ctx"] is not None
  assert seen["ctx"].event_log is None


@pytest.mark.parametrize(
  ("session_cache_approved", "automatic_approval_reason"),
  [
    (True, None),
    (False, "headless hook allowed"),
    (False, None),
  ],
)
def test_fresh_owner_lifecycle_blocks_every_automatic_source(
  tmp_path: Path,
  session_cache_approved: bool,
  automatic_approval_reason: str | None,
) -> None:
  class AutoPolicy:
    policy_id = "auto-policy"
    policy_version = "1"

    def __init__(self) -> None:
      self.decide_calls = 0
      self.resolved: list[str] = []

    async def decide(self, **_kwargs: Any) -> PolicyApprovalDecision:
      self.decide_calls += 1
      return PolicyApprovalDecision(
        outcome="auto_approve",
        reason="custom automatic policy",
        allow_persistent_grant=True,
        persistent_grant_scope_hint="state_write:merge_diligence_pr",
        grant_reference="grant-should-not-survive",
      )

    async def on_resolve(self, *, request: Any) -> None:
      self.resolved.append(request.approval_id)

  async def fail_if_prompted(*_args: Any, **_kwargs: Any) -> None:
    raise AssertionError("headless fresh-owner lifecycle must not prompt")

  store = SQLiteApprovalStore(tmp_path / "fresh-owner.sqlite3")
  policy = AutoPolicy()
  result = asyncio.run(
    lifecycle_helpers.run_approval_lifecycle(
      store=store,
      policy=policy,
      session=SimpleNamespace(),
      tool_call_id="promotion-call",
      tool_name="merge_diligence_pr",
      tool_input={"confirm_merge": True},
      qualifier="",
      reason="promotion",
      allow_persistent=True,
      approval_constraint="fresh_human_owner",
      required_owner_user_id="owner-1",
      session_cache_approved=session_cache_approved,
      automatic_approval_reason=automatic_approval_reason,
      deny_user_prompt=True,
      resolve_run_context_fn=lambda: RunContext(
        user_id="owner-1",
        request_id="request-1",
        decider_role="owner",
      ),
      current_skill_fn=lambda: None,
      redact_for_approval_request_fn=lambda *_args: ({}, "args-hash"),
      resolve_tool_class_fn=lambda _tool_name: "state_write",
      effective_trade_approval_decision_fn=lambda _name, _args, decision: decision,
      await_user_approval_via_pending_tools_fn=fail_if_prompted,
      approval_queue_timeout_seconds_fn=lambda _expiry: 1.0,
    )
  )

  request = result["request"]
  assert result["approved"] is False
  assert result["allow_tool_type"] is False
  assert request.state == "auto_denied"
  assert request.authorization_mode == "HUMAN"
  assert request.grant_reference is None
  assert request.persistent_grant_scope is None
  assert request.approval_constraint == "fresh_human_owner"
  assert request.required_owner_user_id == "owner-1"
  assert policy.decide_calls == 1
  assert policy.resolved == [request.approval_id]
