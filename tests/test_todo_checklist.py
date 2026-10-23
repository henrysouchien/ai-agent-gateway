from __future__ import annotations

from typing import Any

import pytest

from agent_gateway.todo_checklist import (
  EMPTY_CHECKLIST,
  TodoChecklist,
  TodoError,
  apply_analyst_edit,
  apply_model_op,
  build_todo_snapshot_payload,
  describe_change,
  fold_todo_snapshots,
  render_checklist_projection,
  render_open_items_reminder,
)


def _init(phases=None, *, base=None):
  phases = phases or [
    {"name": "Research", "items": ["Reconcile revenue", "Customer concentration"]},
    {"name": "Synthesis", "items": ["Assess thesis"]},
  ]
  return apply_model_op(EMPTY_CHECKLIST, {"op": "init", "phases": phases}, base_revision=base)


def _statuses(checklist: TodoChecklist) -> list[tuple[str, str]]:
  return [(item.content, item.status) for _, item in checklist.items()]


def test_init_assigns_ids_and_activates_first_pending() -> None:
  checklist = _init()
  assert checklist.revision == 1
  assert _statuses(checklist) == [
    ("Reconcile revenue", "in_progress"),
    ("Customer concentration", "pending"),
    ("Assess thesis", "pending"),
  ]
  ids = [item.id for _, item in checklist.items()]
  assert len(set(ids)) == 3 and all(i.startswith("it_") for i in ids)


def test_done_advances_active_pointer_and_skips_blocked() -> None:
  checklist = _init()
  checklist = apply_model_op(
    checklist,
    {"op": "block", "item": "Customer concentration", "reason": "Awaiting analyst: segment or geo?"},
    base_revision=checklist.revision,
  )
  checklist = apply_model_op(checklist, {"op": "done", "item": "Reconcile revenue"}, base_revision=checklist.revision)
  assert _statuses(checklist) == [
    ("Reconcile revenue", "completed"),
    ("Customer concentration", "blocked"),
    ("Assess thesis", "in_progress"),
  ]
  assert checklist.revision == 3
  assert [item.blocker for _, item in checklist.items()][1] == "Awaiting analyst: segment or geo?"


def test_start_moves_the_single_active_item() -> None:
  checklist = apply_model_op(_init(), {"op": "start", "item": "Assess thesis"}, base_revision=1)
  assert _statuses(checklist) == [
    ("Reconcile revenue", "pending"),
    ("Customer concentration", "pending"),
    ("Assess thesis", "in_progress"),
  ]


def test_done_phase_and_done_all_mark_only_open_items() -> None:
  checklist = apply_model_op(_init(), {"op": "drop", "item": "Customer concentration"}, base_revision=1)
  checklist = apply_model_op(checklist, {"op": "done", "phase": "Research"}, base_revision=2)
  assert _statuses(checklist)[:2] == [("Reconcile revenue", "completed"), ("Customer concentration", "abandoned")]
  checklist = apply_model_op(checklist, {"op": "done", "all": True}, base_revision=3)
  assert all(status in {"completed", "abandoned"} for _, status in _statuses(checklist))
  assert checklist.open_items() == []


def test_unblock_returns_to_pending_and_rm_removes() -> None:
  checklist = _init()
  checklist = apply_model_op(checklist, {"op": "block", "item": "Assess thesis", "reason": "waiting"}, base_revision=1)
  checklist = apply_model_op(checklist, {"op": "unblock", "item": "Assess thesis"}, base_revision=2)
  assert dict(_statuses(checklist))["Assess thesis"] == "pending"
  checklist = apply_model_op(checklist, {"op": "rm", "items": ["Assess thesis"]}, base_revision=3)
  assert [content for content, _ in _statuses(checklist)] == ["Reconcile revenue", "Customer concentration"]


def test_append_creates_phase_when_absent() -> None:
  checklist = apply_model_op(_init(), {"op": "append", "phase": "Review", "items": ["Second look"]}, base_revision=1)
  assert [phase.name for phase in checklist.phases] == ["Research", "Synthesis", "Review"]
  assert checklist.phases[-1].items[0].content == "Second look"


def test_duplicate_content_and_unknown_item_change_nothing() -> None:
  checklist = _init()
  with pytest.raises(TodoError) as duplicate:
    apply_model_op(checklist, {"op": "append", "phase": "Research", "items": ["Assess thesis"]}, base_revision=1)
  assert duplicate.value.code == "duplicate_content"
  with pytest.raises(TodoError) as unknown:
    apply_model_op(checklist, {"op": "done", "item": "Nope"}, base_revision=1)
  assert unknown.value.code == "unknown_item"
  assert unknown.value.to_dict()["content"] == "Nope"
  with pytest.raises(TodoError) as missing_reason:
    apply_model_op(checklist, {"op": "block", "item": "Assess thesis"}, base_revision=1)
  assert missing_reason.value.code == "invalid_text"
  assert checklist.revision == 1


def test_view_returns_checklist_unchanged() -> None:
  checklist = _init()
  assert apply_model_op(checklist, {"op": "view"}, base_revision=0) is checklist


def test_analyst_edit_keeps_ids_and_describes_change() -> None:
  checklist = _init()
  before = {item.content: item.id for _, item in checklist.items()}
  edited = apply_analyst_edit(
    checklist,
    expected_revision=1,
    phases=[
      {
        "name": "Research",
        "items": [
          {"id": before["Reconcile revenue"], "content": "Reconcile revenue", "status": "completed"},
        ],
      },
      {
        "name": "Synthesis",
        "items": [
          {"id": before["Assess thesis"], "content": "Assess the thesis", "status": "pending"},
          {"content": "Compare against consensus"},
        ],
      },
    ],
  )
  assert edited.revision == 2
  assert edited.analyst_revision == 2
  assert edited.analyst_change == (
    "marked 'Reconcile revenue' completed; reworded 'Assess thesis' to 'Assess the thesis'; "
    "added 'Compare against consensus'; removed 'Customer concentration'"
  )
  by_content = {item.content: item for _, item in edited.items()}
  assert by_content["Assess the thesis"].id == before["Assess thesis"]
  assert by_content["Assess the thesis"].status == "in_progress"


def test_analyst_edit_with_stale_revision_is_rejected_with_current_checklist() -> None:
  checklist = _init()
  with pytest.raises(TodoError) as stale:
    apply_analyst_edit(checklist, expected_revision=0, phases=[])
  assert stale.value.code == "stale_revision"
  assert stale.value.to_dict()["checklist"]["revision"] == 1


def test_model_mutation_after_analyst_edit_is_rejected_until_it_rebases() -> None:
  checklist = _init()
  edited = apply_analyst_edit(
    checklist,
    expected_revision=1,
    phases=[{"name": "Research", "items": [{"content": "Only this"}]}],
  )
  # The model formed `done all` at revision 1; the analyst moved to 2.
  with pytest.raises(TodoError) as stale:
    apply_model_op(edited, {"op": "done", "all": True}, base_revision=1)
  assert stale.value.code == "stale_revision"
  assert stale.value.to_dict()["analyst_change"] == edited.analyst_change
  assert _statuses(edited) == [("Only this", "in_progress")]
  # Rebased on the analyst's revision, the same op applies.
  rebased = apply_model_op(edited, {"op": "done", "all": True}, base_revision=2)
  assert _statuses(rebased) == [("Only this", "completed")]
  # The agent's own consecutive ops at one base never conflict.
  again = apply_model_op(rebased, {"op": "append", "phase": "Research", "items": ["More"]}, base_revision=2)
  assert again.revision == 4


def test_fold_takes_latest_snapshot_scoped_to_the_run() -> None:
  first = _init()
  second = apply_model_op(first, {"op": "done", "item": "Reconcile revenue"}, base_revision=1)
  events = [
    {"type": "attach", "gateway_session_id": "sess_a"},
    build_todo_snapshot_payload(first, actor="agent", op="init", gateway_session_id="sess_a"),
    build_todo_snapshot_payload(second, actor="agent", op="done", gateway_session_id="sess_a"),
    build_todo_snapshot_payload(_init([{"name": "Other", "items": ["x"]}]), actor="agent", op="init", gateway_session_id="sess_b"),
  ]
  unscoped = fold_todo_snapshots(events)
  assert unscoped.revision == 2 and dict(_statuses(unscoped))["Reconcile revenue"] == "completed"
  assert fold_todo_snapshots(events, gateway_session_id="sess_a").revision == 2
  assert [c for c, _ in _statuses(fold_todo_snapshots(events, gateway_session_id="sess_b"))] == ["x"]
  assert fold_todo_snapshots(events, gateway_session_id="sess_c") is EMPTY_CHECKLIST
  assert fold_todo_snapshots([]) is EMPTY_CHECKLIST


def test_snapshot_round_trips_through_payload() -> None:
  checklist = apply_model_op(_init(), {"op": "block", "item": "Assess thesis", "reason": "why"}, base_revision=1)
  payload = build_todo_snapshot_payload(checklist, actor="agent", op="block", gateway_session_id="s")
  assert payload["type"] == "todo_snapshot" and payload["revision"] == 2
  restored = TodoChecklist.from_payload(payload)
  assert restored == checklist


def test_renderers_name_open_items_and_the_bound() -> None:
  checklist = apply_model_op(_init(), {"op": "done", "item": "Reconcile revenue"}, base_revision=1)
  reminder = render_open_items_reminder(checklist, sent=2, bound=3)
  assert reminder.startswith("You stopped with 2 checklist items still open:")
  assert "Reconcile revenue" not in reminder
  assert "- Research\n  - [>] Customer concentration" in reminder
  assert reminder.endswith("(Reminder 2 of 3.)")
  projection = render_checklist_projection(checklist)
  assert projection.startswith("Current checklist (revision 2)")
  assert "[x] Reconcile revenue" in projection


def test_describe_change_reports_reorder_only() -> None:
  checklist = _init([{"name": "P", "items": ["a", "b"]}])
  ids = {item.content: item.id for _, item in checklist.items()}
  reordered = apply_analyst_edit(
    checklist,
    expected_revision=1,
    phases=[{"name": "P", "items": [
      {"id": ids["b"], "content": "b", "status": "pending"},
      {"id": ids["a"], "content": "a", "status": "pending"},
    ]}],
  )
  assert describe_change(checklist, reordered) == "reordered items"

async def _run(coro: Any) -> Any:
  import asyncio

  return asyncio.get_event_loop_policy().new_event_loop().run_until_complete(coro)


def test_todo_patch_route_happy_path_stamps_analyst_revision() -> None:
  """PATCH applies against the current revision, appends one durable snapshot
  (actor analyst) and emits the same event; a stale revision is rejected with
  the current checklist and changes nothing."""
  import asyncio
  from types import SimpleNamespace

  from fastapi import HTTPException

  from agent_gateway import todo_checklist
  from agent_gateway import server_chat_control_routes as routes

  session = SimpleNamespace(session_id="s", todo_checklist_state=None)
  state = todo_checklist.ensure_todo_session_state(
    session, gateway_session_id="s", scope_to_run=False
  )
  state.apply_model({"op": "init", "phases": [{"name": "W", "items": ["a", "b"]}]})

  auth = SimpleNamespace(verify_token=lambda token: session)
  appended: list[dict] = []
  emitted: list[dict] = []

  async def append(session_arg: Any, event: dict) -> None:
    appended.append(event)

  body = SimpleNamespace(expected_revision=1, phases=[
    {"name": "W", "items": [
      {"id": list(i.id for _, i in state.checklist.items())[0], "content": "a", "status": "completed"},
    ]},
  ])
  result = asyncio.new_event_loop().run_until_complete(
    routes.chat_todo_patch_response(
      SimpleNamespace(headers={}),  # type: ignore[arg-type]
      body,
      auth=auth,
      get_bearer_token=lambda h: "",
      append_todo_snapshot=append,
      emit_to_session=lambda session_arg, event: emitted.append(event),
    )
  )
  payload = __import__("json").loads(bytes(result.body))
  assert payload["revision"] == 2
  assert len(appended) == 1 and appended[0]["actor"] == "analyst"
  assert len(emitted) == 1 and emitted[0]["op"] == "patch"
  assert state.pending_analyst_note is not None
  assert state.checklist.analyst_revision == 2

  # Stale: the checklist moved to 2; a PATCH made against 1 fails with 409.
  body2 = SimpleNamespace(expected_revision=1, phases=[])
  try:
    asyncio.new_event_loop().run_until_complete(
      routes.chat_todo_patch_response(
        SimpleNamespace(headers={}),  # type: ignore[arg-type]
        body2,
        auth=auth,
        get_bearer_token=lambda h: "",
        append_todo_snapshot=append,
        emit_to_session=lambda session_arg, event: emitted.append(event),
      )
    )
  except HTTPException as exc:
    assert exc.status_code == 409
    assert exc.detail["checklist"]["revision"] == 2  # type: ignore[index]
  else:
    raise AssertionError("stale PATCH must fail")
  assert len(appended) == 1, "a rejected edit persists nothing"


def test_todo_get_route_returns_snapshot_or_empty() -> None:
  import asyncio
  from types import SimpleNamespace

  from agent_gateway import todo_checklist
  from agent_gateway import server_chat_control_routes as routes

  session = SimpleNamespace(session_id="s", todo_checklist_state=None)
  auth = SimpleNamespace(verify_token=lambda token: session)
  result = asyncio.new_event_loop().run_until_complete(
    routes.chat_todo_get_response(
      SimpleNamespace(headers={}),  # type: ignore[arg-type]
      auth=auth,
      get_bearer_token=lambda h: "",
    )
  )
  import json

  import json as _json
  assert _json.loads(result.body)["revision"] == 0  # type: ignore[arg-type]
  state = todo_checklist.ensure_todo_session_state(
    session, gateway_session_id="s", scope_to_run=False
  )
  state.apply_model({"op": "init", "phases": [{"name": "W", "items": ["a"]}]})
  result = asyncio.new_event_loop().run_until_complete(
    routes.chat_todo_get_response(
      SimpleNamespace(headers={}),  # type: ignore[arg-type]
      auth=auth,
      get_bearer_token=lambda h: "",
    )
  )
  assert _json.loads(result.body)["revision"] == 1  # type: ignore[arg-type]


def test_todo_patch_before_session_log_binding_survives_first_chat_load() -> None:
  """An analyst PATCH accepted before the session log is bound must not be
  wiped out when the first /chat turn loads the newly bound log."""
  import asyncio
  import json as _json
  from types import SimpleNamespace
  from agent_gateway import todo_checklist
  from agent_gateway import server_chat_control_routes as routes

  class _FakeLog:
    def __init__(self) -> None:
      self.events: list[dict[str, Any]] = []
    async def query(self, *, event_types: Any = None, order: Any = None, limit: Any = None, cursor: Any = None) -> tuple[list[Any], None]:
      matched = [SimpleNamespace(event=e) for e in self.events if e.get("type") in event_types]
      return matched, None
    async def append(self, event: dict[str, Any]) -> None:
      self.events.append(event)

  session = SimpleNamespace(session_id="sess_pre", todo_checklist_state=None, agent_session_log=None)
  auth = SimpleNamespace(verify_token=lambda token: session)

  # 1. Analyst sends PATCH before /chat bound the log:
  body = SimpleNamespace(expected_revision=0, phases=[
    {"name": "Research", "items": [{"content": "Review 10-K", "status": "pending"}]},
  ])
  async def append(s: Any, ev: Any) -> None:
    pass
  result = asyncio.new_event_loop().run_until_complete(
    routes.chat_todo_patch_response(
      SimpleNamespace(headers={}),  # type: ignore[arg-type]
      body,
      auth=auth,
      get_bearer_token=lambda h: "",
      append_todo_snapshot=append,
      emit_to_session=lambda s, ev: None,
    )
  )
  assert _json.loads(result.body)["revision"] == 1  # type: ignore[arg-type]

  # 2. First /chat turn arrives: log is bound, runner calls state.load(log):
  log = _FakeLog()
  session.agent_session_log = log
  state = session.todo_checklist_state
  assert state is not None
  asyncio.new_event_loop().run_until_complete(state.load(log))

  # The edit must NOT be reset to 0:
  assert state.checklist.revision == 1
  assert [i.content for _, i in state.checklist.items()] == ["Review 10-K"]
  # And it must be persisted to the log so future resumes find it:
  assert len(log.events) == 1
  assert log.events[0]["type"] == "todo_snapshot"
  assert log.events[0]["revision"] == 1


def test_todo_autonomous_run_scoping_does_not_inherit_earlier_run() -> None:
  """Autonomous runs sharing a profile log must only fold their own run's snapshots."""
  import asyncio
  from types import SimpleNamespace
  from agent_gateway import todo_checklist
  from agent_gateway.runner_run_loop import RunnerRunLoopMixin

  class _AutonomousRunner(RunnerRunLoopMixin):
    def __init__(self, session_id: str, log: Any) -> None:
      self._role = "writer"
      self._soft_budget_kind = "autonomous"
      self._gateway_session_id = session_id
      self._gateway_session = SimpleNamespace(session_id=session_id, todo_checklist_state=None)
      self._agent_session_log = log

  class _SharedLog:
    def __init__(self) -> None:
      self.events: list[dict[str, Any]] = []
    async def query(self, *, event_types: Any = None, order: Any = None, limit: Any = None, cursor: Any = None) -> tuple[list[Any], None]:
      matched = [SimpleNamespace(event=e) for e in self.events if e.get("type") in event_types]
      return matched, None
    async def append(self, event: dict[str, Any]) -> None:
      self.events.append(event)

  shared_log = _SharedLog()

  # Run 1 creates a checklist with open items under run 1 via runner._load_todo_run_state():
  runner1 = _AutonomousRunner("run_1", shared_log)
  state1 = asyncio.new_event_loop().run_until_complete(runner1._load_todo_run_state())
  assert state1 is not None
  state1.apply_model({"op": "init", "phases": [{"name": "A", "items": ["Open task"]}]})
  asyncio.new_event_loop().run_until_complete(
    shared_log.append(state1.snapshot_event(actor="agent", op="init"))
  )

  # Run 2 starts on the same shared log via runner._load_todo_run_state():
  runner2 = _AutonomousRunner("run_2", shared_log)
  state2 = asyncio.new_event_loop().run_until_complete(runner2._load_todo_run_state())
  assert state2 is not None

  # Run 2 must start with an empty checklist and zero open items:
  assert state2.checklist.revision == 0
  assert state2.checklist.open_items() == []
  assert state2.reminder_due() is False



class _JournalLog:
  """A session journal fake: typed query, newest-first on ``desc``, paged by ``limit``."""

  def __init__(self, events: list[dict[str, Any]]) -> None:
    self.events = events
    self.queries: list[dict[str, Any]] = []

  async def query(
    self, *, event_types: Any = None, order: Any = "asc", limit: Any = None, cursor: Any = None,
  ) -> tuple[list[Any], Any]:
    from types import SimpleNamespace

    self.queries.append({"event_types": set(event_types or ()), "order": order, "limit": limit})
    matched = [SimpleNamespace(event=e) for e in self.events if e.get("type") in event_types]
    if order == "desc":
      matched.reverse()
    if limit is None:
      return matched, None
    start = cursor or 0
    next_cursor = start + limit if start + limit < len(matched) else None
    return matched[start:start + limit], next_cursor

  async def append(self, event: dict[str, Any]) -> None:
    self.events.append(event)


def _user(text: Any, **extra: Any) -> dict[str, Any]:
  return {"type": "user_message", "content": text, "client_kind": "cli", **extra}


def _skill(name: str, **extra: Any) -> dict[str, Any]:
  return {"type": "tool_call_start", "tool_name": "invoke_skill", "tool_input": {"skill_name": name}, **extra}


def _load(state: Any, log: Any) -> None:
  import asyncio

  asyncio.run(state.load(log))


def _state(session_id: str = "sess_todo") -> Any:
  from agent_gateway import todo_checklist

  return todo_checklist.TodoSessionState(gateway_session_id=session_id, scope_to_run=False)


def _as_skill(name: str):
  from agent_gateway.skill_context import set_current_skill
  from agent_gateway.skill_limits import ActiveSkillAdmission, SkillExecutionLimits

  return set_current_skill(
    ActiveSkillAdmission(skill_name=name, execution_limits=SkillExecutionLimits(max_tokens=None))
  )


_CRITICAL_FACTORS_PHASES = [
  {"name": "Phase 1 — candidate assembly", "items": ["Assemble candidates"]},
  {"name": "Phase 2 — rank/test/validate", "items": ["Score I*P*F"]},
  {"name": "Phase 3 — verdict + FMS", "items": ["Emit fms_propose_critical_factors"]},
]


def test_nested_skill_init_nests_before_open_outer_phases() -> None:
  from agent_gateway.skill_context import reset_current_skill

  root = _init([
    {"name": "Recon", "items": ["Collect filings"]},
    {"name": "Model build", "items": ["Persist ScenarioSet", "Build model"]},
  ])
  root = apply_model_op(root, {"op": "done", "item": "Collect filings"}, base_revision=1)
  token = _as_skill("critical-factors")
  try:
    nested = apply_model_op(
      root,
      {
        "op": "init",
        "_invoked_skills": ("build-model", "critical-factors"),
        "phases": [
          {"name": "Phase 1 — candidate assembly", "items": ["Enumerate candidates"]},
          {"name": "Phase 3 — verdict + FMS", "items": ["Propose critical factors"]},
        ],
      },
      base_revision=root.revision,
    )
  finally:
    reset_current_skill(token)

  assert [p.name for p in nested.phases] == [
    "Recon",
    "Phase 1 — candidate assembly",
    "Phase 3 — verdict + FMS",
    "Model build",
  ]
  assert dict(_statuses(nested))["Enumerate candidates"] == "in_progress"
  assert dict(_statuses(nested))["Persist ScenarioSet"] == "pending"

  nested = apply_model_op(nested, {"op": "done", "phase": "Phase 1 — candidate assembly"}, base_revision=nested.revision)
  nested = apply_model_op(nested, {"op": "done", "phase": "Phase 3 — verdict + FMS"}, base_revision=nested.revision)
  assert dict(_statuses(nested))["Persist ScenarioSet"] == "in_progress"
  assert len(nested.open_items()) == 2


def test_phase_addressing_is_exact_or_a_unique_skill_qualified_suffix() -> None:
  checklist = _init([
    {"name": "Phase 1 — candidate assembly", "items": ["A"]},
    {"name": "critical-factors: Research", "items": ["B"]},
    {"name": "Research", "items": ["C"]},
  ])
  with pytest.raises(TodoError) as excinfo:
    apply_model_op(checklist, {"op": "done", "phase": "Phase 1"}, base_revision=checklist.revision)
  assert excinfo.value.code == "unknown_phase"
  exact = apply_model_op(checklist, {"op": "done", "phase": "Research"}, base_revision=checklist.revision)
  assert dict(_statuses(exact)) == {"A": "in_progress", "B": "pending", "C": "completed"}

  qualified_only = _init([
    {"name": "critical-factors: Research", "items": ["B"]},
    {"name": "Writing", "items": ["D"]},
  ])
  done = apply_model_op(qualified_only, {"op": "done", "phase": "Research"}, base_revision=qualified_only.revision)
  assert dict(_statuses(done)) == {"B": "completed", "D": "in_progress"}

  ambiguous = _init([
    {"name": "a: Research", "items": ["B"]},
    {"name": "b: Research", "items": ["D"]},
  ])
  with pytest.raises(TodoError) as ambiguous_error:
    apply_model_op(ambiguous, {"op": "done", "phase": "Research"}, base_revision=ambiguous.revision)
  assert ambiguous_error.value.code == "unknown_phase"


def test_payc_nested_skill_keeps_root_open_with_a_generic_continuation_phase() -> None:
  from agent_gateway.skill_context import reset_current_skill

  log = _JournalLog([
    _user("Build me a model for PAYC.\n\n## Current Session Context\n```json\n{}\n```"),
    _skill("build-model"),
    _skill("business-model-construction"),
    _skill("critical-factors"),
    _skill("critical-factors", sub_agent_id="child-1"),
  ])
  state = _state("sess_payc")
  _load(state, log)
  assert state.root_deliverable == "Build me a model for PAYC."
  assert state.invoked_skills == ("build-model", "business-model-construction", "critical-factors")

  token = _as_skill("critical-factors")
  try:
    state.apply_model({"op": "init", "phases": _CRITICAL_FACTORS_PHASES})
  finally:
    reset_current_skill(token)
  assert [p.name for p in state.checklist.phases] == [
    "Phase 1 — candidate assembly",
    "Phase 2 — rank/test/validate",
    "Phase 3 — verdict + FMS",
    "Root — build-model continuation",
  ]
  for phase in _CRITICAL_FACTORS_PHASES:
    state.apply_model({"op": "done", "phase": phase["name"]})
  assert state.reminder_due() is True
  [(phase, item)] = state.checklist.open_items()
  assert phase.name == "Root — build-model continuation"
  assert item.content == "Complete root deliverable: Build me a model for PAYC."
  assert item.status == "in_progress"


def test_nested_init_does_not_add_a_continuation_the_incoming_items_already_name() -> None:
  from agent_gateway.skill_context import reset_current_skill

  state = _state()
  state.root_deliverable = "Build me a model for PAYC."
  state.invoked_skills = ("build-model", "critical-factors")
  token = _as_skill("critical-factors")
  try:
    state.apply_model({
      "op": "init",
      "phases": [{"name": "Stage", "items": ["Emit factors", "Return to build-model chain"]}],
    })
  finally:
    reset_current_skill(token)
  assert [p.name for p in state.checklist.phases] == ["Stage"]


def test_single_skill_turn_adds_no_continuation_phase() -> None:
  from agent_gateway.skill_context import reset_current_skill

  log = _JournalLog([
    _user("Explain Paycom's business model in two paragraphs."),
    _skill("company-overview"),
  ])
  state = _state()
  _load(state, log)
  assert state.root_deliverable == "Explain Paycom's business model in two paragraphs."
  assert state.invoked_skills == ("company-overview",)

  token = _as_skill("company-overview")
  try:
    state.apply_model({"op": "init", "phases": [{"name": "Research", "items": ["Read the 10-K"]}]})
  finally:
    reset_current_skill(token)
  assert [p.name for p in state.checklist.phases] == ["Research"]
  assert _statuses(state.checklist) == [("Read the 10-K", "in_progress")]


def test_new_user_turn_resets_root_deliverable_and_invoked_skills() -> None:
  from agent_gateway.skill_context import reset_current_skill

  log = _JournalLog([
    _user("Build me a model for PAYC."),
    _skill("build-model"),
    _skill("critical-factors"),
    _user("Now write an earnings memo for ADP.\n\n## Current Session Context\n```json\n{}\n```"),
    _skill("earnings-memo"),
  ])
  state = _state()
  _load(state, log)
  assert state.root_deliverable == "Now write an earnings memo for ADP."
  assert state.invoked_skills == ("earnings-memo",)

  token = _as_skill("earnings-memo")
  try:
    state.apply_model({"op": "init", "phases": [{"name": "Memo", "items": ["Draft ADP memo"]}]})
  finally:
    reset_current_skill(token)
  assert [p.name for p in state.checklist.phases] == ["Memo"]


def test_nudge_and_operator_messages_do_not_start_a_turn() -> None:
  log = _JournalLog([
    _user("Build me a model for PAYC."),
    _skill("build-model"),
    _user("[History was compacted into the summary above. Continue the task from that summary.]"),
    _user("Continue from where you left off."),
    _user("operator note", client_kind="operator"),
    _skill("critical-factors"),
  ])
  state = _state()
  _load(state, log)
  assert state.root_deliverable == "Build me a model for PAYC."
  assert state.invoked_skills == ("build-model", "critical-factors")


def test_load_refreshes_the_turn_each_call_but_folds_snapshots_once() -> None:
  events: list[dict[str, Any]] = [_user("Build me a model for PAYC.")]
  log = _JournalLog(events)
  state = _state()
  _load(state, log)
  assert state.invoked_skills == ()
  snapshot_queries = [q for q in log.queries if q["event_types"] == {"todo_snapshot"}]
  assert len(snapshot_queries) == 1

  state.checklist = _init([{"name": "A", "items": ["Keep"]}])
  events.extend([_skill("build-model"), _skill("critical-factors")])
  _load(state, log)

  assert state.invoked_skills == ("build-model", "critical-factors")
  assert [q for q in log.queries if q["event_types"] == {"todo_snapshot"}] == snapshot_queries
  assert _statuses(state.checklist) == [("Keep", "in_progress")]


def test_turn_scan_stops_at_the_latest_user_message() -> None:
  events: list[dict[str, Any]] = [_user("Old request"), *[_skill(f"old-{i}") for i in range(200)]]
  events.extend([_user("Current request"), _skill("current-skill")])
  log = _JournalLog(events)
  state = _state()
  _load(state, log)
  assert state.root_deliverable == "Current request"
  assert state.invoked_skills == ("current-skill",)
  assert [q for q in log.queries if q["order"] == "desc"] == [
    {"event_types": {"user_message", "tool_call_start"}, "order": "desc", "limit": 64},
  ]


def test_turn_scan_skips_runner_nudges_and_stops_at_an_attachment_only_message() -> None:
  from agent_gateway.runner_run_loop_defaults import MAX_TOKENS_NUDGE

  events: list[dict[str, Any]] = [
    _user("Old request"),
    _skill("old-skill"),
    _user([{"type": "image", "source": {"type": "base64", "data": "x"}}]),
    _skill("current-skill"),
    _user(MAX_TOKENS_NUDGE),
    _user("[System: background result delivered]"),
  ]
  state = _state()
  _load(state, _JournalLog(events))
  assert state.root_deliverable is None
  assert state.invoked_skills == ("current-skill",)

  state = _state()
  _load(state, _JournalLog([*events[:2], _user("Current request", attachments=[{"name": "a.png"}])]))
  assert state.root_deliverable == "Current request"
  assert state.invoked_skills == ()


def test_root_init_replaces_the_checklist_on_replan() -> None:
  from agent_gateway.skill_context import reset_current_skill

  log = _JournalLog([_user("Build me a model for PAYC."), _skill("build-model")])
  state = _state()
  _load(state, log)
  state.apply_model({"op": "init", "phases": [{"name": "Plan A", "items": ["Pull PAYC 10-K", "Build DCF"]}]})

  state.apply_model({"op": "init", "phases": [{"name": "Plan B", "items": ["Write ADP memo"]}]})
  assert [p.name for p in state.checklist.phases] == ["Plan B"]
  assert _statuses(state.checklist) == [("Write ADP memo", "in_progress")]

  # Any top-level skill (build-model, critical-factors, etc.) re-planning is the root
  # agent's own plan, not a nested stage:
  for skill_name in ("build-model", "critical-factors"):
    log_skill = _JournalLog([_user("Rank critical factors for PAYC."), _skill(skill_name)])
    st = _state()
    _load(st, log_skill)
    token = _as_skill(skill_name)
    try:
      st.apply_model({"op": "init", "phases": [{"name": "Plan A", "items": ["Pull 10-K"]}]})
      st.apply_model({"op": "init", "phases": [{"name": "Plan C", "items": ["Persist scenarios"]}]})
    finally:
      reset_current_skill(token)
    assert [p.name for p in st.checklist.phases] == ["Plan C"]

def test_skill_context_parent_skill_and_first_skill_with_open_root_checklist_nest() -> None:
  from agent_gateway.skill_context import current_parent_skill, reset_current_skill

  t_outer = _as_skill("build-model")
  try:
    assert current_parent_skill() is None
    root = _init([{"name": "Root", "items": ["Collect filings", "Build model"]}])
    t_inner = _as_skill("historical-foundation")
    try:
      assert current_parent_skill() == "build-model"
      # Even without _invoked_skills in op, parent_skill from skill_context marks this nested:
      nested = apply_model_op(
        root,
        {"op": "init", "phases": [{"name": "HFF", "items": ["Persist foundation"]}]},
        base_revision=root.revision,
      )
      assert [p.name for p in nested.phases] == ["HFF", "Root"]
    finally:
      reset_current_skill(t_inner)
  finally:
    reset_current_skill(t_outer)
  assert current_parent_skill() is None
