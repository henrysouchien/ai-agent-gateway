"""The root agent's working checklist: one state owner for every writer.

``docs/design/agent-todo-primitive-design-2026-10-08.md`` §3-§6. The
checklist is bookkeeping about intent: phases of items with a status, an
optional blocker, and a server-assigned id the model never sees. Every
accepted change is one ``todo_snapshot`` journal event; the current state is
the fold of those events (``fold_todo_snapshots``). Model operations address
items by exact content; analyst edits replace the phases wholesale and keep
ids for the items they retain.

Two writers share one revision space. A model operation carries the revision
the model last saw and is rejected (``stale_revision``) when an analyst
snapshot newer than that exists, so a model ``init`` formed before an analyst
edit cannot erase it. An analyst edit carries the revision it was made
against and is rejected when the checklist moved. Neither rejection changes
state; both return the current checklist so the writer can redo its change.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import secrets
from typing import Any, Callable, Iterable, Literal, Mapping, Sequence

TodoStatus = Literal["pending", "in_progress", "completed", "abandoned", "blocked"]
TodoActor = Literal["agent", "analyst"]

TODO_SNAPSHOT_EVENT_TYPE = "todo_snapshot"
TODO_STATUSES: tuple[TodoStatus, ...] = (
  "pending",
  "in_progress",
  "completed",
  "abandoned",
  "blocked",
)
OPEN_STATUSES: frozenset[str] = frozenset({"pending", "in_progress"})
MODEL_OPS: tuple[str, ...] = (
  "init",
  "append",
  "start",
  "done",
  "drop",
  "block",
  "unblock",
  "rm",
  "view",
)
TODO_CONTENT_MAX_CHARS = 500
TODO_BLOCKER_MAX_CHARS = 500
TODO_PHASE_MAX_CHARS = 120


class TodoError(Exception):
  """Input validation at the state owner; nothing changes when raised."""

  def __init__(self, code: str, message: str, **fields: Any) -> None:
    super().__init__(message)
    self.code = code
    self.message = message
    self.fields = dict(fields)

  def to_dict(self) -> dict[str, Any]:
    return {"code": self.code, "message": self.message, **self.fields}


@dataclass(frozen=True, slots=True)
class TodoItem:
  id: str
  content: str
  status: TodoStatus = "pending"
  blocker: str | None = None

  def to_payload(self) -> dict[str, Any]:
    payload: dict[str, Any] = {
      "id": self.id,
      "content": self.content,
      "status": self.status,
    }
    if self.blocker is not None:
      payload["blocker"] = self.blocker
    return payload


@dataclass(frozen=True, slots=True)
class TodoPhase:
  name: str
  items: tuple[TodoItem, ...] = ()

  def to_payload(self) -> dict[str, Any]:
    return {"name": self.name, "items": [item.to_payload() for item in self.items]}


@dataclass(frozen=True, slots=True)
class TodoChecklist:
  revision: int = 0
  phases: tuple[TodoPhase, ...] = ()
  # Revision of the latest analyst snapshot and a one-line description of it;
  # the model-mutation precondition reads both (§4 concurrency).
  analyst_revision: int = 0
  analyst_change: str | None = None

  def items(self) -> list[tuple[TodoPhase, TodoItem]]:
    return [(phase, item) for phase in self.phases for item in phase.items]

  def open_items(self) -> list[tuple[TodoPhase, TodoItem]]:
    return [(phase, item) for phase, item in self.items() if item.status in OPEN_STATUSES]

  def counts(self) -> dict[str, int]:
    counts = {status: 0 for status in TODO_STATUSES}
    for _, item in self.items():
      counts[item.status] += 1
    return counts

  def is_empty(self) -> bool:
    return not any(phase.items for phase in self.phases)

  def to_payload(self) -> dict[str, Any]:
    payload: dict[str, Any] = {
      "revision": self.revision,
      "phases": [phase.to_payload() for phase in self.phases],
      "counts": self.counts(),
    }
    if self.analyst_revision:
      payload["analyst_revision"] = self.analyst_revision
    if self.analyst_change is not None:
      payload["analyst_change"] = self.analyst_change
    return payload

  @classmethod
  def from_payload(cls, payload: Mapping[str, Any]) -> "TodoChecklist":
    phases: list[TodoPhase] = []
    for raw_phase in payload.get("phases") or ():
      items = tuple(
        TodoItem(
          id=str(raw["id"]),
          content=str(raw["content"]),
          status=_status(raw.get("status", "pending")),
          blocker=raw.get("blocker"),
        )
        for raw in raw_phase.get("items") or ()
      )
      phases.append(TodoPhase(name=str(raw_phase["name"]), items=items))
    return cls(
      revision=int(payload.get("revision") or 0),
      phases=tuple(phases),
      analyst_revision=int(payload.get("analyst_revision") or 0),
      analyst_change=payload.get("analyst_change"),
    )


EMPTY_CHECKLIST = TodoChecklist()


def _status(value: Any) -> TodoStatus:
  if value not in TODO_STATUSES:
    raise TodoError("invalid_status", f"unknown status {value!r}", status=value)
  return value


def _text(value: Any, *, what: str, limit: int) -> str:
  if not isinstance(value, str) or not value.strip():
    raise TodoError("invalid_text", f"{what} must be non-empty text", **{what: value})
  text = value.strip()
  if len(text) > limit:
    raise TodoError("invalid_text", f"{what} exceeds {limit} characters", **{what: text[:40]})
  return text


def _new_id(existing: set[str]) -> str:
  while True:
    candidate = "it_" + secrets.token_hex(4)
    if candidate not in existing:
      existing.add(candidate)
      return candidate


def _normalize_active(phases: Sequence[TodoPhase]) -> tuple[TodoPhase, ...]:
  """At most one item in progress; else the first pending becomes active."""
  active_seen = False
  out: list[TodoPhase] = []
  for phase in phases:
    items: list[TodoItem] = []
    for item in phase.items:
      if item.status == "in_progress":
        if active_seen:
          item = replace(item, status="pending")
        active_seen = True
      items.append(item)
    out.append(replace(phase, items=tuple(items)))
  if active_seen:
    return tuple(out)
  for phase_index, phase in enumerate(out):
    for item_index, item in enumerate(phase.items):
      if item.status == "pending":
        items = list(phase.items)
        items[item_index] = replace(item, status="in_progress")
        out[phase_index] = replace(phase, items=tuple(items))
        return tuple(out)
  return tuple(out)


def _check_unique(phases: Sequence[TodoPhase]) -> None:
  seen_content: set[str] = set()
  seen_ids: set[str] = set()
  seen_phases: set[str] = set()
  for phase in phases:
    if phase.name in seen_phases:
      raise TodoError("duplicate_phase", f"phase {phase.name!r} appears twice", phase=phase.name)
    seen_phases.add(phase.name)
    for item in phase.items:
      if item.content in seen_content:
        raise TodoError(
          "duplicate_content",
          f"item {item.content!r} already exists",
          content=item.content,
        )
      if item.id in seen_ids:
        raise TodoError("duplicate_id", f"item id {item.id!r} appears twice", id=item.id)
      seen_content.add(item.content)
      seen_ids.add(item.id)


def _find_item(checklist: TodoChecklist, content: str) -> tuple[int, int]:
  for phase_index, phase in enumerate(checklist.phases):
    for item_index, item in enumerate(phase.items):
      if item.content == content:
        return phase_index, item_index
  raise TodoError("unknown_item", f"no checklist item {content!r}", content=content)


def _find_phase(checklist: TodoChecklist, name: str) -> int:
  for index, phase in enumerate(checklist.phases):
    if phase.name == name:
      return index
  raise TodoError("unknown_phase", f"no checklist phase {name!r}", phase=name)


def _with_item(
  phases: Sequence[TodoPhase],
  phase_index: int,
  item_index: int,
  update: Callable[[TodoItem], TodoItem],
) -> list[TodoPhase]:
  out = list(phases)
  items = list(out[phase_index].items)
  items[item_index] = update(items[item_index])
  out[phase_index] = replace(out[phase_index], items=tuple(items))
  return out


def _settle(
  checklist: TodoChecklist,
  phases: Sequence[TodoPhase],
) -> TodoChecklist:
  normalized = _normalize_active(phases)
  _check_unique(normalized)
  return replace(checklist, revision=checklist.revision + 1, phases=normalized)


def _mark(
  checklist: TodoChecklist,
  op: Mapping[str, Any],
  status: TodoStatus,
) -> TodoChecklist:
  """``done`` / ``drop``: one item, one phase, or everything still open."""

  def mark(item: TodoItem) -> TodoItem:
    if item.status in ("completed", "abandoned"):
      return item
    return replace(item, status=status, blocker=None)

  if op.get("item") is not None:
    content = _text(op["item"], what="item", limit=TODO_CONTENT_MAX_CHARS)
    phase_index, item_index = _find_item(checklist, content)
    return _settle(checklist, _with_item(checklist.phases, phase_index, item_index, mark))
  if op.get("phase") is not None:
    phase_index = _find_phase(checklist, _text(op["phase"], what="phase", limit=TODO_PHASE_MAX_CHARS))
    phases = list(checklist.phases)
    phases[phase_index] = replace(
      phases[phase_index], items=tuple(mark(item) for item in phases[phase_index].items)
    )
    return _settle(checklist, phases)
  if op.get("all") is True:
    phases = [replace(phase, items=tuple(mark(item) for item in phase.items)) for phase in checklist.phases]
    return _settle(checklist, phases)
  raise TodoError("invalid_op", f"{op.get('op')} needs item, phase, or all=true")


def _parse_phases(
  raw_phases: Any,
  *,
  existing_ids: set[str],
  keep_ids: bool,
) -> list[TodoPhase]:
  if not isinstance(raw_phases, Sequence) or isinstance(raw_phases, str):
    raise TodoError("invalid_op", "phases must be a list of {name, items}")
  phases: list[TodoPhase] = []
  for raw_phase in raw_phases:
    if not isinstance(raw_phase, Mapping):
      raise TodoError("invalid_op", "each phase must be {name, items}")
    name = _text(raw_phase.get("name"), what="phase", limit=TODO_PHASE_MAX_CHARS)
    items: list[TodoItem] = []
    for raw in raw_phase.get("items") or ():
      if isinstance(raw, str):
        items.append(TodoItem(id=_new_id(existing_ids), content=_text(raw, what="item", limit=TODO_CONTENT_MAX_CHARS)))
        continue
      if not isinstance(raw, Mapping):
        raise TodoError("invalid_op", "each item must be text or {content, status, blocker}")
      content = _text(raw.get("content"), what="item", limit=TODO_CONTENT_MAX_CHARS)
      status = _status(raw.get("status", "pending"))
      blocker = raw.get("blocker")
      if blocker is not None:
        blocker = _text(blocker, what="blocker", limit=TODO_BLOCKER_MAX_CHARS)
      if status == "blocked" and blocker is None:
        raise TodoError("invalid_op", f"blocked item {content!r} needs a blocker", content=content)
      if status != "blocked":
        blocker = None
      raw_id = raw.get("id") if keep_ids else None
      item_id = str(raw_id) if raw_id else _new_id(existing_ids)
      items.append(TodoItem(id=item_id, content=content, status=status, blocker=blocker))
    phases.append(TodoPhase(name=name, items=tuple(items)))
  return phases


def apply_model_op(
  checklist: TodoChecklist,
  op: Mapping[str, Any],
  *,
  base_revision: int | None,
) -> TodoChecklist:
  """Apply one ``todo`` tool operation. ``view`` returns the checklist unchanged.

  ``base_revision`` is the checklist revision the model last saw. A mutation
  is rejected with ``stale_revision`` when an analyst snapshot newer than that
  exists; the error carries the current checklist and the analyst's change.
  """
  name = op.get("op")
  if name not in MODEL_OPS:
    raise TodoError("invalid_op", f"unknown todo op {name!r}", op=name)
  if name == "view":
    return checklist
  if (
    base_revision is not None
    and checklist.analyst_revision
    and checklist.analyst_revision > base_revision
  ):
    raise TodoError(
      "stale_revision",
      "the analyst edited the checklist after the revision you last saw; "
      "redo your change against the current checklist",
      checklist=checklist.to_payload(),
      analyst_change=checklist.analyst_change,
    )
  existing_ids = {item.id for _, item in checklist.items()}
  if name == "init":
    phases = _parse_phases(op.get("phases"), existing_ids=set(), keep_ids=False)
    return _settle(replace(checklist, phases=()), phases)
  if name == "append":
    phase_name = _text(op.get("phase"), what="phase", limit=TODO_PHASE_MAX_CHARS)
    raw_items = op.get("items")
    if not isinstance(raw_items, Sequence) or isinstance(raw_items, str) or not raw_items:
      raise TodoError("invalid_op", "append needs a non-empty items list")
    new_items = tuple(
      TodoItem(id=_new_id(existing_ids), content=_text(raw, what="item", limit=TODO_CONTENT_MAX_CHARS))
      for raw in raw_items
    )
    phases = list(checklist.phases)
    for index, phase in enumerate(phases):
      if phase.name == phase_name:
        phases[index] = replace(phase, items=phase.items + new_items)
        break
    else:
      phases.append(TodoPhase(name=phase_name, items=new_items))
    return _settle(checklist, phases)
  if name in ("start", "block", "unblock"):
    content = _text(op.get("item"), what="item", limit=TODO_CONTENT_MAX_CHARS)
    phase_index, item_index = _find_item(checklist, content)
    if name == "start":
      phases = [
        replace(
          phase,
          items=tuple(
            replace(item, status="pending") if item.status == "in_progress" else item
            for item in phase.items
          ),
        )
        for phase in checklist.phases
      ]
      phases = _with_item(phases, phase_index, item_index, lambda item: replace(item, status="in_progress", blocker=None))
      return _settle(checklist, phases)
    if name == "block":
      reason = _text(op.get("reason"), what="blocker", limit=TODO_BLOCKER_MAX_CHARS)
      phases = _with_item(
        checklist.phases, phase_index, item_index, lambda item: replace(item, status="blocked", blocker=reason)
      )
      return _settle(checklist, phases)
    phases = _with_item(
      checklist.phases, phase_index, item_index,
      lambda item: replace(item, status="pending", blocker=None) if item.status == "blocked" else item,
    )
    return _settle(checklist, phases)
  if name == "done":
    return _mark(checklist, op, "completed")
  if name == "drop":
    return _mark(checklist, op, "abandoned")
  # rm
  raw_items = op.get("items")
  if not isinstance(raw_items, Sequence) or isinstance(raw_items, str) or not raw_items:
    raise TodoError("invalid_op", "rm needs a non-empty items list")
  targets = {_text(raw, what="item", limit=TODO_CONTENT_MAX_CHARS) for raw in raw_items}
  for content in targets:
    _find_item(checklist, content)
  phases = [
    replace(phase, items=tuple(item for item in phase.items if item.content not in targets))
    for phase in checklist.phases
  ]
  return _settle(checklist, phases)


def apply_analyst_edit(
  checklist: TodoChecklist,
  *,
  expected_revision: int,
  phases: Any,
) -> TodoChecklist:
  """Replace the phases from an analyst edit made against ``expected_revision``.

  Items that carry an ``id`` keep it; items without one are new. A stale
  ``expected_revision`` is rejected with the current checklist attached.
  """
  if expected_revision != checklist.revision:
    raise TodoError(
      "stale_revision",
      f"checklist is at revision {checklist.revision}, edit was made against {expected_revision}",
      checklist=checklist.to_payload(),
    )
  known_ids = {item.id for _, item in checklist.items()}
  parsed = _parse_phases(phases, existing_ids=set(known_ids), keep_ids=True)
  for phase in parsed:
    for item in phase.items:
      if item.id.startswith("it_") and item.id not in known_ids and len(item.id) != 11:
        raise TodoError("unknown_item", f"unknown item id {item.id!r}", id=item.id)
  settled = _settle(checklist, parsed)
  change = describe_change(checklist, settled)
  return replace(settled, analyst_revision=settled.revision, analyst_change=change)


def describe_change(before: TodoChecklist, after: TodoChecklist) -> str:
  """One line naming what an edit did, by item id; shown to the model."""
  before_items = {item.id: item for _, item in before.items()}
  after_items = {item.id: item for _, item in after.items()}
  parts: list[str] = []
  for item_id, item in after_items.items():
    prior = before_items.get(item_id)
    if prior is None:
      parts.append(f"added '{item.content}'")
    elif prior.content != item.content:
      parts.append(f"reworded '{prior.content}' to '{item.content}'")
    elif prior.status != item.status and {prior.status, item.status} != OPEN_STATUSES:
      # A pending<->in_progress swap is the active pointer settling, not an edit.
      parts.append(f"marked '{item.content}' {item.status}")
    elif prior.blocker != item.blocker and item.blocker is not None:
      parts.append(f"blocked '{item.content}': {item.blocker}")
  for item_id, prior in before_items.items():
    if item_id not in after_items:
      parts.append(f"removed '{prior.content}'")
  if not parts:
    parts.append("reordered items")
  return "; ".join(parts)


def build_todo_snapshot_payload(
  checklist: TodoChecklist,
  *,
  actor: TodoActor,
  op: str,
  gateway_session_id: str,
) -> dict[str, Any]:
  """The durable ``todo_snapshot`` event body (``runner_session_events`` wraps it)."""
  return {
    "type": TODO_SNAPSHOT_EVENT_TYPE,
    "gateway_session_id": gateway_session_id,
    "actor": actor,
    "op": op,
    "revision": checklist.revision,
    "phases": [phase.to_payload() for phase in checklist.phases],
    "analyst_revision": checklist.analyst_revision,
    "analyst_change": checklist.analyst_change,
  }


def fold_todo_snapshots(
  events: Iterable[Mapping[str, Any]],
  *,
  gateway_session_id: str | None = None,
) -> TodoChecklist:
  """Latest ``todo_snapshot`` wins. ``gateway_session_id`` scopes the fold to one
  run (autonomous and scheduled logs are shared across runs, §5)."""
  latest: Mapping[str, Any] | None = None
  for event in events:
    if event.get("type") != TODO_SNAPSHOT_EVENT_TYPE:
      continue
    if gateway_session_id is not None and event.get("gateway_session_id") != gateway_session_id:
      continue
    if latest is None or int(event.get("revision") or 0) >= int(latest.get("revision") or 0):
      latest = event
  if latest is None:
    return EMPTY_CHECKLIST
  return TodoChecklist.from_payload(latest)


def render_checklist(checklist: TodoChecklist, *, open_only: bool = False) -> str:
  lines: list[str] = []
  for phase in checklist.phases:
    items = [item for item in phase.items if not open_only or item.status in OPEN_STATUSES]
    if not items:
      continue
    lines.append(f"- {phase.name}")
    for item in items:
      marker = {
        "pending": "[ ]",
        "in_progress": "[>]",
        "completed": "[x]",
        "abandoned": "[-]",
        "blocked": "[!]",
      }[item.status]
      suffix = f" (blocked: {item.blocker})" if item.status == "blocked" and item.blocker else ""
      lines.append(f"  - {marker} {item.content}{suffix}")
  return "\n".join(lines)


def render_checklist_projection(checklist: TodoChecklist) -> str:
  """The post-compaction 'Current checklist' message (§5)."""
  return (
    f"Current checklist (revision {checklist.revision}); address items by their exact text:\n"
    + render_checklist(checklist)
  )


def render_open_items_reminder(checklist: TodoChecklist, *, sent: int, bound: int) -> str:
  """The stop-time reminder (§6). ``sent`` counts this reminder."""
  open_items = checklist.open_items()
  noun = "item" if len(open_items) == 1 else "items"
  return (
    f"You stopped with {len(open_items)} checklist {noun} still open:\n"
    + render_checklist(checklist, open_only=True)
    + "\n\nContinue them, or update the checklist: mark items done if they are finished, "
    "abandoned if they no longer apply, or blocked with the reason if you are waiting on "
    f"the analyst or on data you cannot get. (Reminder {sent} of {bound}.)"
  )


def render_analyst_edit_note(checklist: TodoChecklist) -> str:
  """The one-line note the runner adds after an analyst edit lands mid-turn (§8)."""
  change = checklist.analyst_change or "changed the checklist"
  return (
    f"The analyst edited the checklist: {change}. Do not re-add removed items unless asked. "
    f"Current revision {checklist.revision}."
  )


TODO_REMINDER_BOUND = 3


class TodoSessionState:
  """The live checklist of one session, shared by the tool handler, the run
  loop and the analyst ``PATCH`` door.

  ``model_seen_revision`` is the revision the model last had in context (a
  ``todo`` result, the post-compaction projection, a reminder, or an analyst
  edit note); every model mutation is checked against it. ``reminders_sent``
  and ``moved_since_reminder`` are the stop-time bound (§6) and live only
  with the running turn: ``reset_turn`` runs at each analyst turn start.
  ``pending_analyst_note`` is set by an analyst edit while a turn runs and
  consumed by the runner at its next request boundary.
  """

  def __init__(
    self,
    checklist: TodoChecklist = EMPTY_CHECKLIST,
    *,
    gateway_session_id: str,
    scope_to_run: bool,
  ) -> None:
    self.checklist = checklist
    self.gateway_session_id = gateway_session_id
    # Autonomous and scheduled runs share one log across runs; their fold
    # takes only this run's snapshots (§5).
    self.scope_to_run = scope_to_run
    self.loaded = False
    self.model_seen_revision = checklist.revision
    self.reminders_sent = 0
    self.moved_since_reminder = True
    self.pending_analyst_note: str | None = None

  async def load(self, agent_session_log: Any | None) -> TodoChecklist:
    """Fold the checklist from the journal once per state object."""
    if self.loaded:
      return self.checklist
    if agent_session_log is None:
      return self.checklist
    entries, _ = await agent_session_log.query(
      event_types={TODO_SNAPSHOT_EVENT_TYPE},
      order="asc",
    )
    folded = fold_todo_snapshots(
      (entry.event for entry in entries),
      gateway_session_id=self.gateway_session_id if self.scope_to_run else None,
    )
    if self.checklist.revision > folded.revision:
      await agent_session_log.append(self.snapshot_event(actor="analyst", op="patch"))
    else:
      self.checklist = folded
      self.model_seen_revision = self.checklist.revision
    self.loaded = True
    return self.checklist

  def snapshot_event(self, *, actor: TodoActor, op: str) -> dict[str, Any]:
    return build_todo_snapshot_payload(
      self.checklist,
      actor=actor,
      op=op,
      gateway_session_id=self.gateway_session_id,
    )

  def apply_model(self, op: Mapping[str, Any]) -> TodoChecklist:
    """Apply one tool op against the revision the model last saw.

    A ``stale_revision`` rejection carries the current checklist back as the
    tool result, so the model has seen that revision once the error is in
    its context; its redo applies against it.
    """
    try:
      updated = apply_model_op(self.checklist, op, base_revision=self.model_seen_revision)
    except TodoError as exc:
      if exc.code == "stale_revision":
        self.model_seen_revision = self.checklist.revision
      raise
    if updated is not self.checklist:
      self.checklist = updated
      self.moved_since_reminder = True
    self.model_seen_revision = self.checklist.revision
    return self.checklist

  def apply_analyst(self, *, expected_revision: int, phases: Any) -> TodoChecklist:
    self.checklist = apply_analyst_edit(
      self.checklist, expected_revision=expected_revision, phases=phases
    )
    self.moved_since_reminder = True
    self.pending_analyst_note = render_analyst_edit_note(self.checklist)
    return self.checklist

  def take_analyst_note(self) -> str | None:
    """The mid-turn edit note, once; marks its revision as seen by the model."""
    note = self.pending_analyst_note
    self.pending_analyst_note = None
    if note is not None:
      self.model_seen_revision = self.checklist.revision
    return note

  def mark_projected(self) -> None:
    """The checklist text just entered model context (projection/reminder)."""
    self.model_seen_revision = self.checklist.revision

  def reset_turn(self) -> None:
    self.reminders_sent = 0
    self.moved_since_reminder = True

  def reminder_due(self) -> bool:
    """Open items, within the bound, and movement since the last reminder."""
    return (
      bool(self.checklist.open_items())
      and self.reminders_sent < TODO_REMINDER_BOUND
      and self.moved_since_reminder
    )

  def next_reminder(self) -> str:
    self.reminders_sent += 1
    self.moved_since_reminder = False
    self.mark_projected()
    return render_open_items_reminder(
      self.checklist, sent=self.reminders_sent, bound=TODO_REMINDER_BOUND
    )

  def note_tool_call(self) -> None:
    """A tool call since the last reminder re-arms it (§6 bound)."""
    self.moved_since_reminder = True

  def measurements(self) -> dict[str, int]:
    return {
      "todo_reminders_sent": self.reminders_sent,
      "todo_open_at_end": len(self.checklist.open_items()),
    }


def todo_session_state(session: Any) -> TodoSessionState | None:
  """The session's live checklist state, when a root runtime declared one."""
  state = getattr(session, "todo_checklist_state", None)
  return state if isinstance(state, TodoSessionState) else None


def ensure_todo_session_state(
  session: Any,
  *,
  gateway_session_id: str,
  scope_to_run: bool,
) -> TodoSessionState:
  state = todo_session_state(session)
  if state is None or state.gateway_session_id != gateway_session_id:
    state = TodoSessionState(gateway_session_id=gateway_session_id, scope_to_run=scope_to_run)
    session.todo_checklist_state = state
  elif state.scope_to_run != scope_to_run:
    state.scope_to_run = scope_to_run
  return state


__all__ = [
  "EMPTY_CHECKLIST",
  "MODEL_OPS",
  "OPEN_STATUSES",
  "TODO_REMINDER_BOUND",
  "TODO_SNAPSHOT_EVENT_TYPE",
  "TODO_STATUSES",
  "TodoActor",
  "TodoChecklist",
  "TodoError",
  "TodoItem",
  "TodoPhase",
  "TodoSessionState",
  "TodoStatus",
  "apply_analyst_edit",
  "apply_model_op",
  "build_todo_snapshot_payload",
  "describe_change",
  "ensure_todo_session_state",
  "fold_todo_snapshots",
  "render_analyst_edit_note",
  "render_checklist",
  "render_checklist_projection",
  "render_open_items_reminder",
  "todo_session_state",
]
