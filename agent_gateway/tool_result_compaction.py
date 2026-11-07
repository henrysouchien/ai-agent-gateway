from __future__ import annotations

import json
import os
import uuid
from typing import Any, Callable, Dict

from agent_workflow_contracts.context_refs import (
  project_persist_envelope_delta,
  project_report_envelope_receipt,
)
from .autonomous_event_channel import autonomous_event_fits_frame
from .tool_result_spill import SpillPublication, SpillSink, normalize_spill_sink, write_spill_set
from .tool_result_semantics import status_error_has_detail
from hank_model_transport.policy_imports import load_server_policy_module

MODEL_TOOL_RESULT_MAX_CHARS = 60_000
MODEL_TOOL_RESULT_MAX_CHARS_ENV = "AGENT_GATEWAY_MAX_MODEL_TOOL_RESULT_CHARS"
MODEL_TOOL_RESULT_MIN_CHARS = 4_000
SPILL_TRUNCATED_TOOL_RESULTS_ENV = "AGENT_GATEWAY_SPILL_TRUNCATED_TOOL_RESULTS"


def model_tool_result_max_chars() -> int:
  raw = os.getenv(MODEL_TOOL_RESULT_MAX_CHARS_ENV)
  if raw is None or not raw.strip():
    return MODEL_TOOL_RESULT_MAX_CHARS
  try:
    value = int(raw)
  except ValueError:
    return MODEL_TOOL_RESULT_MAX_CHARS
  if value <= 0:
    return 0
  return max(MODEL_TOOL_RESULT_MIN_CHARS, value)


def spill_truncated_tool_results_enabled() -> bool:
  raw = os.getenv(SPILL_TRUNCATED_TOOL_RESULTS_ENV)
  if raw is None:
    return True
  return raw.strip().lower() not in {"0", "false", "no"}


_ELIDED_ITEMS_KEY = "_elided_items"
_ELIDED_KEYS_KEY = "_elided_keys"
_ELIDED_DEPTH_KEY = "_elided_depth"
_ELIDED_CHARS_MARKER = "...<elided chars="
_PROJECTION_MAX_DEPTH = 12

# A navigation row keeps its scalars under this cap on the ordinary rungs: the
# row is the producer's index entry, so its key and value-space fields are what
# a caller navigates by.  Only the model ladder's last-resort rungs narrow it.
_NAVIGATION_ROW_STRING_CHARS = 160

# Successively harder elision settings: (max list items, max string chars, max dict
# entries).  The first setting whose serialized payload fits the model budget wins,
# so a result loses bulk before it loses shape.
_PROJECTION_LADDER = (
  (200, 8_000, 400),
  (100, 4_000, 400),
  (50, 2_000, 400),
  (25, 1_000, 300),
  (12, 500, 200),
  (6, 300, 120),
  (3, 200, 80),
  (1, 120, 48),
  (0, 80, 32),
  (0, 40, 16),
)

# The model ladder adds two navigation caps to each rung: the string cap of a
# compact navigation row and how many items of a preserved navigation container
# survive (None = all).  The existing rungs keep whole containers at 160-char
# rows; only after the bulk rungs are spent does a navigation container itself
# narrow, so an oversized index degrades to a marked prefix instead of
# collapsing the whole result to the omission stub.
_MODEL_PROJECTION_LADDER = (
  *((*rung, _NAVIGATION_ROW_STRING_CHARS, None) for rung in _PROJECTION_LADDER),
  (0, 40, 16, 40, None),
  (0, 40, 16, 40, 512),
  (0, 40, 16, 40, 128),
  (0, 40, 16, 40, 32),
  (0, 40, 16, 40, 0),
)

# The live-stream projection walks the projection ladder at full depth, then ends on
# a rung that keeps the top level only: a result's width and nesting can outrun
# every full-depth rung, and the top level is where an envelope's fields live.
# 400 entries of at most 200 characters is under 330 KB even at four UTF-8
# bytes a character, leaving the frame to the keys and the rest of the event.
_STREAM_PROJECTION_LADDER = (
  *((rung, _PROJECTION_MAX_DEPTH) for rung in _PROJECTION_LADDER),
  ((0, 200, 400), 1),
)


_NAVIGATION_FALLBACK_KEYS = frozenset({"semantic_keys", "required_drivers", "valid_keys"})


def _preserve_navigation_key(key: str) -> bool:
  """Whether the product policy names ``key`` as a producer navigation index."""

  policy = load_server_policy_module()
  preserve = getattr(policy, "preserve_tool_result_navigation_key", None)
  if callable(preserve):
    return bool(preserve(key))
  return key in _NAVIGATION_FALLBACK_KEYS


def _project_navigation_row(
  row: Dict[Any, Any],
  *,
  max_items: int,
  max_entries: int,
  depth: int,
  max_depth: int,
  row_chars: int = _NAVIGATION_ROW_STRING_CHARS,
  max_navigation_items: int | None = None,
) -> Dict[str, Any]:
  """One index row of an over-long navigation list: every scalar, no bulk.

  ``depth`` is the row's own depth.  Scalars (identity, unit, latest value,
  status) survive under a short string cap; nested prose and lists are cut and
  counted, except a nested navigation container, which is projected whole.
  """

  projected: Dict[str, Any] = {}
  elided = 0
  for key, item in row.items():
    if isinstance(item, (dict, list)) and not _preserve_navigation_key(str(key)):
      elided += 1
      continue
    projected[str(key)] = _project_value(
      item,
      max_items=max_items,
      max_string_chars=row_chars,
      max_entries=max_entries,
      depth=depth + 1,
      max_depth=max_depth,
      preserve_list_items=True,
      nav_row_chars=row_chars,
      max_navigation_items=max_navigation_items,
    )
  if elided:
    projected[_ELIDED_KEYS_KEY] = elided
  return projected


def _project_value(
  value: Any,
  *,
  max_items: int,
  max_string_chars: int,
  max_entries: int,
  depth: int,
  max_depth: int = _PROJECTION_MAX_DEPTH,
  preserve_list_items: bool = False,
  nav_row_chars: int = _NAVIGATION_ROW_STRING_CHARS,
  max_navigation_items: int | None = None,
) -> Any:
  """Return ``value`` reshaped to the given elision settings, marking every cut.

  Bulk (long strings, long lists, wide dicts, deep nesting) is dropped; scalars and
  small subtrees are carried through unchanged so a caller can trust an unmarked
  value.  This decides how *much* of a result survives, never which parts matter,
  with one exception: a container the product policy names as a navigation index
  keeps every key and every list item on every ordinary rung, because an elided
  index entry is a key the caller can no longer name; only the model ladder's
  last-resort rungs narrow it to a marked prefix rather than lose the outline.
  """

  if isinstance(value, str):
    if len(value) <= max_string_chars:
      return value
    # Product-minted locators must survive projection whole, never truncated.
    policy = load_server_policy_module()
    preserve = getattr(policy, "preserve_tool_result_string", None)
    if preserve is not None and preserve(value):
      return value
    return f"{value[:max_string_chars]}{_ELIDED_CHARS_MARKER}{len(value) - max_string_chars}>"
  if isinstance(value, (int, float, bool)) or value is None:
    return value
  if isinstance(value, dict):
    if depth >= max_depth:
      return {_ELIDED_DEPTH_KEY: "dict", _ELIDED_KEYS_KEY: len(value)}
    entries = list(value.items())
    selected = entries[:max_entries]
    if len(entries) > max_entries:
      selected.extend(
        (key, item)
        for key, item in entries[max_entries:]
        if _preserve_navigation_key(str(key))
      )
    projected: Dict[str, Any] = {}
    for key, item in selected:
      projected[str(key)] = _project_value(
        item,
        max_items=max_items,
        max_string_chars=max_string_chars,
        max_entries=max_entries,
        depth=depth + 1,
        max_depth=max_depth,
        preserve_list_items=preserve_list_items or _preserve_navigation_key(str(key)),
        nav_row_chars=nav_row_chars,
        max_navigation_items=max_navigation_items,
      )
    if len(entries) > len(selected):
      projected[_ELIDED_KEYS_KEY] = len(entries) - len(selected)
    return projected
  if isinstance(value, list):
    if depth >= max_depth:
      return [{_ELIDED_DEPTH_KEY: "list", _ELIDED_ITEMS_KEY: len(value)}]
    if preserve_list_items:
      compact_rows = len(value) > max_items
      preserved_items = value
      if max_navigation_items is not None and len(value) > max_navigation_items:
        preserved_items = value[:max_navigation_items]
      kept: list[Any] = []
      for item in preserved_items:
        if compact_rows and isinstance(item, dict):
          kept.append(_project_navigation_row(
            item,
            max_items=max_items,
            max_entries=max_entries,
            depth=depth + 1,
            max_depth=max_depth,
            row_chars=nav_row_chars,
            max_navigation_items=max_navigation_items,
          ))
          continue
        kept.append(_project_value(
          item,
          max_items=max_items,
          max_string_chars=max(max_string_chars, nav_row_chars),
          max_entries=max_entries,
          depth=depth + 1,
          max_depth=max_depth,
          preserve_list_items=True,
          nav_row_chars=nav_row_chars,
          max_navigation_items=max_navigation_items,
        ))
      if len(preserved_items) < len(value):
        kept.append({_ELIDED_ITEMS_KEY: len(value) - len(preserved_items)})
      return kept
    items = [
      _project_value(
        item,
        max_items=max_items,
        max_string_chars=max_string_chars,
        max_entries=max_entries,
        depth=depth + 1,
        max_depth=max_depth,
      )
      for item in value[:max_items]
    ]
    if len(value) > max_items:
      items.append({_ELIDED_ITEMS_KEY: len(value) - max_items})
    return items
  return _project_value(
    str(value),
    max_items=max_items,
    max_string_chars=max_string_chars,
    max_entries=max_entries,
    depth=depth,
    max_depth=max_depth,
  )


def _fit_content_projection(
  value: Any,
  *,
  payload: Dict[str, Any],
  max_chars: int,
) -> str | None:
  """Serialize ``payload`` with the most generous projection of ``value`` that fits."""

  # A result that is one large string (or mostly one) should still fill the budget,
  # so the first rung sizes its string cap from the space the envelope leaves free.
  text_budget = max_chars - len(json.dumps(payload, default=str)) - 64
  ladder = _MODEL_PROJECTION_LADDER
  if text_budget > _PROJECTION_LADDER[0][1]:
    first = _MODEL_PROJECTION_LADDER[0]
    ladder = ((first[0], text_budget, first[2], first[3], first[4]), *ladder)

  for max_items, max_string_chars, max_entries, nav_row_chars, max_navigation_items in ladder:
    payload["content_projection"] = _project_value(
      value,
      max_items=max_items,
      max_string_chars=max_string_chars,
      max_entries=max_entries,
      depth=0,
      nav_row_chars=nav_row_chars,
      max_navigation_items=max_navigation_items,
    )
    serialized = json.dumps(payload, default=str)
    if len(serialized) <= max_chars:
      return serialized
  payload.pop("content_projection", None)
  return None



def annotate_result(result: Any, tool_name: str = "") -> Any:
  """Add _runner_warning to generic results with detectable anomalies."""
  if not isinstance(result, dict):
    return result

  warnings: list[str] = []
  interceptor_warnings = result.pop("_interceptor_warnings", None)
  if isinstance(interceptor_warnings, list):
    for warning in interceptor_warnings:
      warnings.append(f"Policy warning: {warning}")

  low_match = result.get("low_match_warning")
  if low_match:
    warnings.append(f"Low match rate detected: {low_match}")

  if tool_name == "run_agent":
    sub_warning = result.get("warning")
    if sub_warning:
      warnings.append(f"Sub-agent warning: {sub_warning}")

  if _status_error_without_detail(result):
    tool_label = f"Tool {tool_name}" if tool_name else "Tool"
    warnings.append(
      f"{tool_label} returned status=error without error detail; "
      "do not retry unchanged input unless required context changed or there is new evidence the failure was transient."
    )

  if not warnings:
    return result

  enriched = dict(result)
  enriched["_runner_warning"] = " | ".join(warnings)
  if low_match:
    enriched["_runner_warning_detail"] = str(low_match)
  return enriched


def _status_error_without_detail(result: dict[str, Any]) -> bool:
  status = result.get("status")
  return isinstance(status, str) and status.strip().lower() == "error" and not status_error_has_detail(result)


def is_error_tool_result_entry(result_entry: Dict[str, Any], content: str) -> bool:
  if result_entry.get("is_error") or "error" in result_entry:
    return True
  try:
    parsed = json.loads(content)
  except Exception:
    return False
  # Only a *truthy* top-level "error" marks a genuine error result. A successful
  # data payload that merely carries `"error": null` (or empty) must still spill;
  # the authoritative tool-error signal is the is_error flag checked above.
  return isinstance(parsed, dict) and bool(parsed.get("error"))


def write_tool_result_spill(
  *,
  work_dir: str,
  tool_name: str,
  tool_use_id: Any,
  content: str,
  uuid_factory=None,
) -> tuple[str, str]:
  sink = normalize_spill_sink(lambda: work_dir)
  assert sink is not None
  publication = write_spill_set(
    sink=sink,
    tool_name=tool_name,
    tool_use_id=tool_use_id,
    content=content,
    model_max_chars=model_tool_result_max_chars(),
    uuid_factory=uuid_factory or uuid.uuid4,
  )
  return publication.filename, publication.abspath


def truncate_model_tool_result_content(
  content: str,
  *,
  tool_name: str,
  max_chars: int,
  spill_ref: str | None = None,
  spill_summary: dict[str, Any] | None = None,
) -> tuple[str, bool]:
  if max_chars <= 0 or len(content) <= max_chars:
    return content, False

  parsed: Any | None = None
  try:
    parsed = json.loads(content)
  except Exception:
    parsed = None

  payload: Dict[str, Any] = {
    "_runner_truncated": True,
    "tool_name": tool_name,
    "original_chars": len(content),
    "message": (
      "The full tool result was retained in the gateway event log. This model-bound "
      "copy keeps the result's own shape and elides only bulk: every elision is marked "
      f"in place with {_ELIDED_ITEMS_KEY}, {_ELIDED_KEYS_KEY}, {_ELIDED_DEPTH_KEY}, or "
      f"'{_ELIDED_CHARS_MARKER}'. Unmarked values are complete and may be reported as-is. "
      "Narrow the tool query (filters, pagination, fewer fields) if you need the elided "
      "detail. If a spill_ref is listed below, use tool_result_read for the full result."
    ),
  }
  if spill_ref is not None:
    payload["spill_ref"] = spill_ref
    if spill_summary:
      payload["spill_summary"] = spill_summary

  projected = _fit_content_projection(
    parsed if parsed is not None else content,
    payload=payload,
    max_chars=max_chars,
  )
  if projected is not None:
    return projected, True

  fallback_payload = {
    "_runner_truncated": True,
    "tool_name": tool_name,
    "original_chars": len(content),
    "message": "Tool result omitted from model context because it exceeded the configured payload limit.",
  }
  if spill_ref is not None:
    fallback_payload["spill_ref"] = spill_ref
    if spill_summary:
      fallback_payload["spill_summary"] = spill_summary
  return json.dumps(fallback_payload, default=str), True


def _publication_summary(publication: SpillPublication) -> dict[str, Any]:
  return {
    "payload_kind": publication.payload_kind,
    "member_count": publication.member_count,
    "total_chars": publication.total_chars,
    "largest_members": list(publication.largest_members),
  }


def _fms_terminal_entry(
  entry: Dict[str, Any],
  content: str,
  *,
  tool_name: str,
) -> Dict[str, Any] | None:
  """The semantic receipt that stands for a settled FMS door result.

  ``fms_persist_*`` keeps the delta receipt (identity, refs, undo handle).
  ``fms_report_*`` keeps only the analyst-facing outcome: filesystem paths and
  citation-machinery instructions stay on the published artifact, never in the
  text the model copies into the answer.
  """
  if is_error_tool_result_entry(entry, content):
    return None
  if tool_name.startswith("fms_persist_"):
    project = project_persist_envelope_delta
  elif tool_name.startswith("fms_report_"):
    project = project_report_envelope_receipt
  else:
    return None
  try:
    parsed_dict = json.loads(content)
  except (TypeError, ValueError):
    return None
  if not isinstance(parsed_dict, dict):
    return None
  delta = project(parsed_dict)
  if delta is None:
    return None
  projected_entry = dict(entry)
  projected_entry["content"] = json.dumps(delta, default=str)
  return projected_entry


def _plain_compacted_entry(
  entry: Dict[str, Any],
  content: str,
  *,
  tool_name: str,
  max_chars: int,
) -> Dict[str, Any]:
  """``entry`` itself when it fits ``max_chars``, else a copy with a plain preview."""
  compacted_content, was_truncated = truncate_model_tool_result_content(
    content,
    tool_name=tool_name,
    max_chars=max_chars,
  )
  if not was_truncated:
    return entry
  compacted = dict(entry)
  compacted["content"] = compacted_content
  return compacted


def compact_model_tool_result_entry(
  result_entry: Dict[str, Any],
  *,
  tool_name: str,
  spill_sink: Callable[[], str] | SpillSink | None = None,
  spill_dir_provider: Callable[[], str] | SpillSink | None = None,
  log_session_id: str,
  logger: Any,
  uuid_factory: Callable[[], Any] | None = None,
  on_spill_publication: Callable[[SpillPublication], None] | None = None,
  max_chars: int | None = None,
  retained_entry: Dict[str, Any] | None = None,
) -> tuple[Dict[str, Any], Dict[str, Any]]:
  """Project one tool result entry for model context, spilling what it elides.

  `max_chars` is the bound the projected content must fit; it defaults to the
  model's tool-result limit, and a delivery channel with a smaller inline
  bound passes that bound.

  Returns `(live_entry, durable_entry)`. `live_entry` is the model-visible
  entry, projected from `result_entry` exactly as before. `durable_entry` is
  what the event log and streams retain.

  `retained_entry` is the boundary projection of `result_entry` (the same
  envelope, its content already through the secret boundary). When given, the
  durable entry, its preview and every spill file are produced from it, and
  the model/stream cuts are taken only after that projection, so no cut can
  leave a secret's prefix. `result_entry` is then the live source only: it is
  neither retained nor spilled, and neither entry is mutated. When omitted the
  two are the same object and the durable entry is the live entry.
  """
  retained = result_entry if retained_entry is None else retained_entry
  content = result_entry.get("content")
  if not isinstance(content, str):
    return result_entry, retained
  retained_content = retained["content"]

  live_terminal = _fms_terminal_entry(result_entry, content, tool_name=tool_name)
  if live_terminal is not None:
    durable_terminal = (
      live_terminal
      if retained is result_entry
      else _fms_terminal_entry(retained, retained_content, tool_name=tool_name)
    )
    if durable_terminal is None:
      durable_terminal = _plain_compacted_entry(
        retained,
        retained_content,
        tool_name=tool_name,
        max_chars=model_tool_result_max_chars() if max_chars is None else max_chars,
      )
    compacted_chars = len(live_terminal["content"])
    logger.info(
      "[%s] Tool %s result projected for model context | original_chars=%d projected_chars=%d",
      log_session_id,
      tool_name,
      len(content),
      compacted_chars,
      extra={
        "data": {
          "event": "tool_result_semantically_compacted",
          "session_id": log_session_id,
          "tool": tool_name,
          "original_chars": len(content),
          "compacted_chars": compacted_chars,
        }
      },
    )
    return live_terminal, durable_terminal

  if max_chars is None:
    max_chars = model_tool_result_max_chars()
  live_entry = _plain_compacted_entry(
    result_entry,
    content,
    tool_name=tool_name,
    max_chars=max_chars,
  )
  # The durable preview is cut from the projected content, never from the live
  # text: a cut taken before the projection can split a secret.
  durable_entry = (
    live_entry
    if retained is result_entry
    else _plain_compacted_entry(
      retained,
      retained_content,
      tool_name=tool_name,
      max_chars=max_chars,
    )
  )
  live_truncated = live_entry is not result_entry
  # Redaction can lengthen content (a short secret becomes the marker), so the
  # retained copy can exceed the limit when the raw one fits. It is then cut
  # with the live entry untouched, and its full projected text still spills.
  durable_truncated = durable_entry is not retained
  if not live_truncated and not durable_truncated:
    return result_entry, durable_entry

  if spill_sink is not None and spill_dir_provider is not None:
    raise ValueError("pass spill_sink or spill_dir_provider, not both")
  configured_sink = normalize_spill_sink(spill_sink if spill_sink is not None else spill_dir_provider)
  if (
    configured_sink is not None
    and spill_truncated_tool_results_enabled()
    and not is_error_tool_result_entry(result_entry, content)
  ):
    try:
      publication = write_spill_set(
        sink=configured_sink,
        tool_name=tool_name,
        tool_use_id=retained.get("tool_use_id"),
        content=retained_content,
        model_max_chars=max_chars,
        uuid_factory=uuid_factory or uuid.uuid4,
      )
      if on_spill_publication is not None:
        on_spill_publication(publication)
      # The model's preview keeps its prior text; only the pointer and summary
      # come from the projected spill set.
      if live_truncated:
        live_compacted_content, _ = truncate_model_tool_result_content(
        content,
        tool_name=tool_name,
        max_chars=max_chars,
        spill_ref=(
          publication.spill_ref
          if configured_sink.capabilities.spill_read
          else None
        ),
        spill_summary=(
          _publication_summary(publication)
          if configured_sink.capabilities.spill_read
          else None
        ),
      )
        live_entry = dict(result_entry)
        live_entry["content"] = live_compacted_content
    except Exception as exc:
      logger.warning(
        "[%s] Tool %s result spill failed; using compacted preview only: %s",
        log_session_id,
        tool_name,
        exc,
        exc_info=True,
      )
  if not live_truncated:
    return live_entry, durable_entry
  logger.info(
    "[%s] Tool %s result compacted for model context | original_chars=%d compacted_chars=%d limit=%d",
    log_session_id,
    tool_name,
    len(content),
    len(str(live_entry.get("content", ""))),
    max_chars,
    extra={
      "data": {
        "event": "tool_result_compacted",
        "session_id": log_session_id,
        "tool": tool_name,
        "original_chars": len(content),
        "compacted_chars": len(str(live_entry.get("content", ""))),
        "limit": max_chars,
      }
    },
  )
  return live_entry, durable_entry


def project_tool_call_complete_for_stream(
  event: Dict[str, Any],
  *,
  result_entry: Dict[str, Any],
  live_entry: Dict[str, Any],
) -> Dict[str, Any]:
  """Return the live-stream projection of one ``tool_call_complete`` event.

  ``event`` arrives already projected through the secret boundary: its
  ``result`` and ``final_tool_result_blocks`` are the retained rendering, and
  the durable event keeps them whole (context rebuilds and resume read the
  durable record).  Every cut below is taken from that projection.
  ``result_entry`` and ``live_entry`` are the *live* model-facing pair from
  ``compact_model_tool_result_entry``: they say whether the model saw a
  compacted copy and supply its spill metadata, and no text of theirs reaches
  the streamed event except the ``spill_ref``/``spill_summary`` that the
  projected spill set minted.  A live stream can be a bounded transport — the
  autonomous event channel refuses any event line over 2 MiB — so an event too
  large for that frame is streamed with its result elided, never as the bulk
  payload the spill set already holds.

  The size question is put to the channel, on this event: the model-facing
  content is a different object, rewritten by the ``on_tool_result`` hooks
  (``code_execute`` images become ``[image: …]`` placeholders there while
  ``event["result"]`` keeps the base64), and it shares the frame with the tool
  input, the durable result blocks and the dispatch metadata.

  The live event is also the in-memory log that skill capture, sub-agent
  result delivery and the ``fms_*`` door extractors read, and those readers
  contract for the result's own envelope.  An event the frame can carry is
  therefore streamed whole regardless of what the model was shown, and one it
  cannot carry keeps the result's own shape with only bulk elided, every cut
  marked in place, typed by ``_runner_truncated``, ``original_chars`` and,
  when the result was spilled, the model copy's ``spill_ref``/``spill_summary``.
  The model ladder is tried first at full depth; when no rung fits, the last
  rung keeps only the top level — at most 400 entries, strings cut at 200
  characters, nested values reduced to depth markers — so the result's values
  stay under 330 KB whatever its width or depth.  Keys and product-minted
  locators pass whole, and the channel still refuses at send an event whose
  input or blocks alone outgrow the frame.  A non-object result, which has no
  envelope to keep, streams under ``content_projection``.

  Returns ``event`` itself when the streamed and durable copies are the same
  object.
  """

  if live_entry is result_entry:
    return event
  if autonomous_event_fits_frame(event):
    return event
  # Compaction only ever returns a distinct entry for a string content, and it
  # writes a JSON object there: the compacted preview or a semantic receipt.
  model_copy = json.loads(live_entry["content"])
  truncation: Dict[str, Any] = {
    "_runner_truncated": True,
    "original_chars": len(result_entry["content"]),
  }
  for key in ("spill_ref", "spill_summary"):
    if key in model_copy:
      truncation[key] = model_copy[key]
  projected = dict(event)
  for (max_items, max_string_chars, max_entries), max_depth in _STREAM_PROJECTION_LADDER:
    value = _project_value(
      event.get("result"),
      max_items=max_items,
      max_string_chars=max_string_chars,
      max_entries=max_entries,
      depth=0,
      max_depth=max_depth,
    )
    projected["result"] = (
      {**value, **truncation}
      if isinstance(value, dict)
      else {**truncation, "content_projection": value}
    )
    if autonomous_event_fits_frame(projected):
      break
  return projected
