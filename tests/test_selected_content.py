from __future__ import annotations

import hashlib

import pytest
from agent_gateway.selected_content import (
  SELECTED_CONTENT_SESSION_MAX_BINDINGS,
  SelectedContentBinding,
  SelectedContentError,
  admit_selected_content_bindings,
  derive_selected_content_name,
  page_selected_content,
  project_selected_content_events,
  render_selected_content_context,
)
from agent_workflow_contracts import (
  ContentHandle,
  OwnerBinding,
  SELECTED_CONTENT_UTF8_CONTRACT,
)


def _binding(name: str, text: str = "hello") -> SelectedContentBinding:
  payload = text.encode("utf-8")
  digest = hashlib.sha256(payload).hexdigest()
  return SelectedContentBinding(
    input_name=name,
    display_name=f"{name}.txt",
    owner=OwnerBinding(tenant_id="tenant-1", session_id="session-1"),
    content=ContentHandle(
      content_id=f"sha256:{digest}",
      content_sha256=digest,
      content_chars=len(text),
      content_bytes=len(payload),
      contract=SELECTED_CONTENT_UTF8_CONTRACT,
      media_type="text/plain",
      encoding="utf-8",
      retention="durable",
    ),
  )


def _sized_binding(name: str, size: int) -> SelectedContentBinding:
  digest = hashlib.sha256(name.encode("utf-8")).hexdigest()
  return SelectedContentBinding(
    input_name=name,
    display_name=f"{name}.txt",
    owner=OwnerBinding(tenant_id="tenant-1", session_id="session-1"),
    content=ContentHandle(
      content_id=f"sha256:{digest}",
      content_sha256=digest,
      content_chars=size,
      content_bytes=size,
      contract=SELECTED_CONTENT_UTF8_CONTRACT,
      media_type="text/plain",
      encoding="utf-8",
      retention="durable",
    ),
  )


def test_name_is_deterministic_and_session_request_position_bound() -> None:
  identity = dict(
    tenant_id="tenant-1",
    session_id="session-1",
    request_id="request-1",
    wire_position=0,
  )
  first = derive_selected_content_name(**identity)

  assert first == derive_selected_content_name(**identity)
  assert first.startswith("selection_")
  assert len(first) == len("selection_") + 24
  assert first != derive_selected_content_name(**{**identity, "wire_position": 1})
  assert first != derive_selected_content_name(**{**identity, "session_id": "session-2"})


def test_replay_treats_legacy_absence_as_empty_and_retries_converge() -> None:
  binding = _binding("selection_0123456789abcdef01234567")
  durable = binding.model_dump(mode="json")

  projection = project_selected_content_events((
    {"type": "user_message", "content": "legacy"},
    {"type": "user_message", "selected_content": [durable]},
    {"type": "user_message", "selected_content": [durable]},
  ))

  assert projection == {binding.input_name: binding}


def test_replay_and_admission_fail_closed_on_conflict_and_limits() -> None:
  name = "selection_0123456789abcdef01234567"
  first = _binding(name, "first")
  second = _binding(name, "second")

  with pytest.raises(SelectedContentError, match="conflicts"):
    project_selected_content_events((
      {"type": "user_message", "selected_content": [first.model_dump(mode="json")]},
      {"type": "user_message", "selected_content": [second.model_dump(mode="json")]},
    ))
  with pytest.raises(SelectedContentError, match="conflicts"):
    admit_selected_content_bindings({name: first}, (second,))

  bindings = tuple(
    _binding(f"selection_{index:024x}", str(index))
    for index in range(SELECTED_CONTENT_SESSION_MAX_BINDINGS + 1)
  )
  with pytest.raises(SelectedContentError) as exc_info:
    admit_selected_content_bindings({}, bindings)
  assert exc_info.value.code == "selected_content_binding_limit"

  oversized = (
    _sized_binding("selection_aaaaaaaaaaaaaaaaaaaaaaaa", 16 * 1024 * 1024),
    _sized_binding("selection_bbbbbbbbbbbbbbbbbbbbbbbb", 16 * 1024 * 1024 + 1),
  )
  with pytest.raises(SelectedContentError) as byte_exc:
    admit_selected_content_bindings({}, oversized)
  assert byte_exc.value.code == "selected_content_byte_limit"


def test_binding_rejects_any_other_content_contract() -> None:
  valid = _binding("selection_0123456789abcdef01234567")
  payload = valid.model_dump(mode="json")
  payload["content"]["contract"] = {
    "namespace": "agent-gateway",
    "name": "other-content",
    "version": "1.0",
    "digest": "sha256:" + ("0" * 64),
  }

  with pytest.raises(ValueError, match="exact UTF-8 contract"):
    SelectedContentBinding.model_validate(payload)


def test_utf8_pages_reassemble_exactly_without_splitting_codepoints() -> None:
  text = ("café 🚀\n" * 7_000) + "terminal"
  after_char = 0
  pages: list[str] = []
  while True:
    page = page_selected_content(text, after_char=after_char)
    pages.append(page.content)
    assert len(page.content.encode("utf-8")) <= 24_000
    assert page.next_after_char == after_char + len(page.content)
    if page.complete:
      break
    after_char = page.next_after_char

  assert "".join(pages) == text


def test_prompt_includes_all_metadata_but_only_budgeted_current_content() -> None:
  current = tuple(
    (_binding(f"selection_{index:024x}", character * 9_000), character * 9_000)
    for index, character in enumerate("ABCDE")
  )
  retained = _binding("selection_ffffffffffffffffffffffff", "RETAINED-PRIVATE")
  visible = {binding.input_name: binding for binding, _text in current}
  visible[retained.input_name] = retained

  rendered = render_selected_content_context(visible, current)

  assert retained.input_name in rendered
  assert "RETAINED-PRIVATE" not in rendered
  assert rendered.count("A") >= 8_000
  assert rendered.count("D") >= 8_000
  assert "E" * 100 not in rendered
