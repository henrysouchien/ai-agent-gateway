from __future__ import annotations

import asyncio
from typing import Any

import pytest
from pydantic import ValidationError

from agent_gateway.approval_policy import RunContext
from agent_gateway.capability_binding import (
  CredentialHandle,
)
from agent_gateway.model_registry import (
  CAPABILITY_IDS,
  INITIAL_MODEL_REGISTRY,
  INITIAL_MODEL_SELECTION_POLICY,
)
from agent_gateway.event_log import EventLog
from agent_gateway.events import UiBlocksReadyEvent, event_from_dict, event_to_dict
from agent_gateway.providers import AnthropicProvider
from agent_gateway.server import MaterializedCredential
from agent_gateway.server_chat_helpers import _dispatch_chat_turn
from agent_gateway.server_models import (
  ChatMessage,
  ChatRequest,
  ChatRuntime,
  ChatTurnInputs,
  UiBlocksContractPin,
)
from agent_gateway.session import GatewaySession
from agent_gateway.tool_dispatcher import ToolDispatcher
from agent_gateway.ui_blocks_run import UiBlocksRunRegistry


_SERVICE_HANDLE = CredentialHandle(
  handle_id="service:ui-blocks-tests:anthropic",
  provider="anthropic",
  principal="service",
  tenant_id="ui-blocks-tests",
  actor_id=None,
)
_PROVIDER = AnthropicProvider()


def _materialize_service_credential(
  handle: CredentialHandle,
) -> MaterializedCredential:
  assert handle is _SERVICE_HANDLE
  return MaterializedCredential(
    handle=handle,
    auth_config={
      "provider": "anthropic",
      "api_key": "test-key",
      "billing_mode": "byok",
      "rate_table_version": "test-v1",
    },
  )


def _configure_runtime_builder(builder) -> None:
  setattr(builder, "_gateway_model_registry", INITIAL_MODEL_REGISTRY)
  setattr(
    builder,
    "_gateway_model_selection_policy",
    INITIAL_MODEL_SELECTION_POLICY,
  )
  setattr(builder, "_gateway_tenant_id", _SERVICE_HANDLE.tenant_id)
  setattr(
    builder,
    "_gateway_service_provider_handles",
    {"anthropic": _SERVICE_HANDLE},
  )
  setattr(
    builder,
    "_gateway_service_auth_config_resolver",
    _materialize_service_credential,
  )
  setattr(
    builder,
    "_gateway_capability_adapter_resolver",
    lambda adapter: _PROVIDER
    if adapter == "anthropic.messages"
    else (_ for _ in ()).throw(ValueError(f"unexpected adapter: {adapter}")),
  )
  setattr(builder, "_gateway_channel_profile_allowlist", None)


def _run(coro):
  return asyncio.run(coro)


async def _accepted(
  registry: UiBlocksRunRegistry,
  ui_blocks_id: str,
  executions: list[str],
  *,
  ready: asyncio.Event | None = None,
  release: asyncio.Event | None = None,
) -> dict[str, Any]:
  reservation = await registry.reserve(ui_blocks_id)
  if not reservation.is_worker:
    return await reservation.wait()
  executions.append(ui_blocks_id)
  if ready is not None:
    ready.set()
  if release is not None:
    await release.wait()

  async def commit(index: int, mark_renamed) -> dict[str, Any]:
    mark_renamed()
    return {"status": "accepted", "ui_blocks_id": ui_blocks_id, "emission_index": index}

  return await reservation.commit(
    commit,
    post_rename_failure=lambda exc, index: {
      "status": "internal_error",
      "ui_blocks_id": ui_blocks_id,
      "emission_index": index,
      "detail": str(exc),
    },
  )


def test_identical_concurrent_calls_execute_and_commit_once() -> None:
  async def case() -> None:
    registry = UiBlocksRunRegistry()
    executions: list[str] = []
    results = await asyncio.gather(
      _accepted(registry, "ub_same", executions),
      _accepted(registry, "ub_same", executions),
    )
    assert executions == ["ub_same"]
    assert results[0] is results[1]
    assert results[0] == {
      "status": "accepted",
      "ui_blocks_id": "ub_same",
      "emission_index": 0,
    }
    assert registry.accepted_emission_count == 1

  _run(case())


def test_concurrent_duplicate_preflight_failure_is_shared_without_hang() -> None:
  async def case() -> None:
    registry = UiBlocksRunRegistry()
    executions = 0
    entered = asyncio.Event()
    release = asyncio.Event()

    async def call() -> dict[str, Any]:
      nonlocal executions
      reservation = await registry.reserve("ub_invalid")
      if not reservation.is_worker:
        return await reservation.wait()
      executions += 1
      entered.set()
      await release.wait()
      return await reservation.validation_failed({
        "status": "validation_failed",
        "failures": [{"block_index": 0, "code": "preflight", "detail": "missing"}],
      })

    worker = asyncio.create_task(call())
    await entered.wait()
    duplicate = asyncio.create_task(call())
    release.set()
    first, second = await asyncio.wait_for(
      asyncio.gather(worker, duplicate),
      timeout=1,
    )
    assert executions == 1
    assert first is second
    assert first["status"] == "validation_failed"

  _run(case())


def test_sequential_deterministic_failure_is_retained() -> None:
  async def case() -> None:
    registry = UiBlocksRunRegistry()
    executions = 0

    async def call() -> dict[str, Any]:
      nonlocal executions
      reservation = await registry.reserve("ub_invalid")
      if not reservation.is_worker:
        return await reservation.wait()
      executions += 1
      return await reservation.validation_failed({
        "status": "validation_failed",
        "failures": [{"block_index": None, "code": "schema", "detail": "bad"}],
      })

    first = await call()
    second = await call()
    assert first is second
    assert executions == 1

  _run(case())


def test_cancelled_duplicate_waiter_does_not_affect_worker_or_other_waiters() -> None:
  async def case() -> None:
    registry = UiBlocksRunRegistry()
    executions: list[str] = []
    ready = asyncio.Event()
    release = asyncio.Event()
    worker = asyncio.create_task(
      _accepted(registry, "ub_same", executions, ready=ready, release=release)
    )
    await ready.wait()
    cancelled_waiter = asyncio.create_task(_accepted(registry, "ub_same", executions))
    surviving_waiter = asyncio.create_task(_accepted(registry, "ub_same", executions))
    await asyncio.sleep(0)
    cancelled_waiter.cancel()
    with pytest.raises(asyncio.CancelledError):
      await cancelled_waiter
    release.set()
    accepted, duplicate = await asyncio.gather(worker, surviving_waiter)
    assert accepted is duplicate
    assert executions == ["ub_same"]

  _run(case())


def test_worker_cancellation_unblocks_waiters_and_removes_pre_rename_entry() -> None:
  async def case() -> None:
    registry = UiBlocksRunRegistry()
    entered = asyncio.Event()

    async def worker_call() -> None:
      reservation = await registry.reserve("ub_cancel")
      assert reservation.is_worker
      entered.set()
      try:
        await asyncio.Event().wait()
      except asyncio.CancelledError as exc:
        await reservation.worker_cancelled(exc)
        raise

    worker = asyncio.create_task(worker_call())
    await entered.wait()
    duplicate = await registry.reserve("ub_cancel")
    waiter = asyncio.create_task(duplicate.wait())
    worker.cancel()
    with pytest.raises(asyncio.CancelledError):
      await worker
    with pytest.raises(asyncio.CancelledError):
      await asyncio.wait_for(waiter, timeout=1)
    assert "ub_cancel" not in registry.entries
    retry = await registry.reserve("ub_cancel")
    assert retry.is_worker

  _run(case())


def test_post_rename_worker_cancellation_retains_identity_and_consumes_index() -> None:
  async def case() -> None:
    registry = UiBlocksRunRegistry()
    renamed = asyncio.Event()
    reservation = await registry.reserve("ub_cancel_committed")

    async def commit(_index: int, mark_renamed) -> dict[str, Any]:
      mark_renamed()
      renamed.set()
      await asyncio.Event().wait()
      raise AssertionError("unreachable")

    worker = asyncio.create_task(
      reservation.commit(
        commit,
        post_rename_failure=lambda _exc, _index: {},
      )
    )
    await renamed.wait()
    duplicate_reservation = asyncio.create_task(
      registry.reserve("ub_cancel_committed")
    )
    await asyncio.sleep(0)
    worker.cancel()
    with pytest.raises(asyncio.CancelledError):
      await worker
    duplicate = await duplicate_reservation
    with pytest.raises(asyncio.CancelledError):
      await duplicate.wait()
    late_duplicate = await registry.reserve("ub_cancel_committed")
    with pytest.raises(asyncio.CancelledError):
      await late_duplicate.wait()
    assert "ub_cancel_committed" in registry.entries
    assert registry.accepted_emission_count == 1

  _run(case())


def test_repeated_worker_cancellation_cannot_strand_waiters() -> None:
  async def case() -> None:
    registry = UiBlocksRunRegistry()
    entered = asyncio.Event()

    async def worker_call() -> None:
      reservation = await registry.reserve("ub_cancel_twice")
      entered.set()
      try:
        await asyncio.Event().wait()
      except asyncio.CancelledError as exc:
        await reservation.worker_cancelled(exc)
        raise

    worker = asyncio.create_task(worker_call())
    await entered.wait()
    duplicate = await registry.reserve("ub_cancel_twice")
    waiter = asyncio.create_task(duplicate.wait())
    await registry._lock.acquire()
    worker.cancel()
    await asyncio.sleep(0)
    worker.cancel()
    registry._lock.release()
    with pytest.raises(asyncio.CancelledError):
      await worker
    with pytest.raises(asyncio.CancelledError):
      await asyncio.wait_for(waiter, timeout=1)
    assert "ub_cancel_twice" not in registry.entries

  _run(case())


def test_pre_rename_store_failure_leaves_no_index_gap() -> None:
  async def case() -> None:
    registry = UiBlocksRunRegistry()
    failed = await registry.reserve("ub_store_failure")

    async def fail_store(_index: int, _mark_renamed) -> dict[str, Any]:
      raise OSError("disk unavailable")

    with pytest.raises(OSError, match="disk unavailable"):
      await failed.commit(
        fail_store,
        post_rename_failure=lambda _exc, _index: {},
      )
    assert registry.accepted_emission_count == 0
    assert "ub_store_failure" not in registry.entries

    result = await _accepted(registry, "ub_next", [])
    assert result["emission_index"] == 0

  _run(case())


def test_post_rename_failure_consumes_index_and_is_retained() -> None:
  async def case() -> None:
    registry = UiBlocksRunRegistry()
    executions = 0
    reservation = await registry.reserve("ub_committed_error")

    async def fail_after_rename(index: int, mark_renamed) -> dict[str, Any]:
      nonlocal executions
      executions += 1
      mark_renamed()
      raise RuntimeError("event append failed")

    first = await reservation.commit(
      fail_after_rename,
      post_rename_failure=lambda exc, index: {
        "status": "internal_error",
        "ui_blocks_id": "ub_committed_error",
        "emission_index": index,
        "detail": str(exc),
      },
    )
    duplicate = await registry.reserve("ub_committed_error")
    second = await duplicate.wait()
    assert first is second
    assert executions == 1
    assert first["emission_index"] == 0
    assert (await _accepted(registry, "ub_next", []))["emission_index"] == 1

  _run(case())


def test_closed_event_log_is_a_retained_post_rename_failure() -> None:
  async def case() -> None:
    registry = UiBlocksRunRegistry()
    event_log = EventLog()
    event_log.close()
    reservation = await registry.reserve("ub_closed_log")

    async def commit(index: int, mark_renamed) -> dict[str, Any]:
      mark_renamed()
      if event_log.append({"type": "ui_blocks_ready"}) is None:
        raise RuntimeError("event log closed before ui_blocks_ready append")
      return {"status": "accepted", "emission_index": index}

    first = await reservation.commit(
      commit,
      post_rename_failure=lambda exc, index: {
        "status": "internal_error",
        "ui_blocks_id": "ub_closed_log",
        "emission_index": index,
        "detail": str(exc),
      },
    )
    duplicate = await registry.reserve("ub_closed_log")
    assert await duplicate.wait() is first
    assert first["status"] == "internal_error"
    assert first["emission_index"] == 0
    assert registry.accepted_emission_count == 1

  _run(case())


def test_distinct_concurrent_payloads_receive_contiguous_indices() -> None:
  async def case() -> None:
    registry = UiBlocksRunRegistry()
    executions: list[str] = []
    results = await asyncio.gather(
      _accepted(registry, "ub_one", executions),
      _accepted(registry, "ub_two", executions),
    )
    assert sorted(result["emission_index"] for result in results) == [0, 1]
    assert sorted(executions) == ["ub_one", "ub_two"]
    assert registry.accepted_emission_count == 2

  _run(case())


class _NullMcp:
  pass


class _TerminalRunner:
  def __init__(self, event_log: EventLog, capability_execution: Any) -> None:
    self._event_log = event_log
    self.capability_execution = capability_execution

  async def run(self, **_kwargs: Any) -> None:
    self._event_log.append({"type": "stream_complete", "usage": {}})


def _session() -> GatewaySession:
  return GatewaySession(
    session_id="sess-ui-blocks",
    api_key_hash="hash",
    created_at=1,
    expires_at=4_000_000_000,
    user_id="alice",
    channel="web",
    tenant_id=_SERVICE_HANDLE.tenant_id,
    allow_service_for_interactive=True,
    model_entitled_capabilities=CAPABILITY_IDS,
    model_entitled_keys=frozenset(INITIAL_MODEL_REGISTRY.models),
  )


async def _no_event(_event: Any) -> None:
  return None


def test_pin_and_run_object_survive_full_gateway_dispatch_chain(tmp_path) -> None:
  async def case() -> None:
    inbound = ChatRequest.model_validate({
      "messages": [{"role": "user", "content": "show me"}],
      "request_id": "caller-controlled-request-id",
      "ui_blocks_contract": {
        "contract_version": 1,
      },
    })
    assert isinstance(inbound.ui_blocks_contract, UiBlocksContractPin)
    inputs = ChatTurnInputs(
      messages=list(inbound.messages),
      request_id=inbound.request_id,
      context=dict(inbound.context),
      metadata=dict(inbound.metadata),
      model_key=inbound.model_key,
      effort=inbound.effort,
      ui_blocks_contract=inbound.ui_blocks_contract,
    )
    captured: dict[str, Any] = {}

    async def build_chat_runtime(*, request, **_kwargs: Any) -> ChatRuntime:
      captured["request"] = request
      capability_bind = request.capability_bind
      assert capability_bind is not None
      run_context = RunContext(
        user_id="alice",
        request_id=str(request.request_id),
        session_id="sess-ui-blocks",
      )
      captured["run_context"] = run_context
      dispatcher = ToolDispatcher(mcp_client=_NullMcp(), run_context=run_context)
      captured["tool_context"] = dispatcher._new_tool_execution_context(
        tool_call_id="call-1",
        tool_name="emit_ui_blocks",
        qualifier="emit_ui_blocks",
        abort_event=None,
        skill_run_id=None,
        step_id=None,
        workspace_dir=None,
        batch_id=None,
      )
      return ChatRuntime(
        system_prompt="test",
        build_runner=lambda event_log, _sid, _started_at: _TerminalRunner(
          event_log,
          request.capability_execution,
        ),
        capability_execution=request.capability_execution,
      )

    _configure_runtime_builder(build_chat_runtime)
    await _dispatch_chat_turn(
      _session(),
      inputs,
      event_log=EventLog(),
      on_event=_no_event,
      build_chat_runtime=build_chat_runtime,
      transcript_dir=tmp_path,
    )

    request = captured["request"]
    ui_run = request._ui_blocks_run
    assert request.ui_blocks_contract is inputs.ui_blocks_contract
    assert ui_run is captured["run_context"].ui_blocks_run
    assert ui_run is captured["tool_context"].ui_blocks_run
    assert ui_run.capability is inputs.ui_blocks_contract
    assert ui_run.session_id == "sess-ui-blocks"
    assert ui_run.turn_key != inbound.request_id

  _run(case())


def test_malformed_contract_pin_is_rejected_without_coercion() -> None:
  with pytest.raises(ValidationError):
    ChatRequest.model_validate({
      "messages": [{"role": "user", "content": "show me"}],
      "ui_blocks_contract": {
        "contract_version": "1",
      },
    })


def test_unpinned_dispatch_has_none_capability_and_turn_keys_are_per_dispatch(tmp_path) -> None:
  async def case() -> None:
    session = _session()
    captured_runs = []

    async def build_chat_runtime(*, request, **_kwargs: Any) -> ChatRuntime:
      captured_runs.append(request._ui_blocks_run)
      capability_bind = request.capability_bind
      assert capability_bind is not None
      return ChatRuntime(
        system_prompt="test",
        build_runner=lambda event_log, _sid, _started_at: _TerminalRunner(
          event_log,
          request.capability_execution,
        ),
        capability_execution=request.capability_execution,
      )

    _configure_runtime_builder(build_chat_runtime)
    for _ in range(2):
      await _dispatch_chat_turn(
        session,
        ChatTurnInputs(
          messages=[ChatMessage(role="user", content="show me")],
          request_id="same-caller-request-id",
          context={},
          metadata={},
          model_key=None,
        ),
        event_log=EventLog(),
        on_event=_no_event,
        build_chat_runtime=build_chat_runtime,
        transcript_dir=tmp_path,
      )
      # Completed turns remain attachable for a grace window; simulate the
      # normal later cleanup before starting the next turn in this unit test.
      session.active_turn = None

    assert [run.capability for run in captured_runs] == [None, None]
    assert captured_runs[0].session_id == captured_runs[1].session_id == session.session_id
    assert captured_runs[0].turn_key != captured_runs[1].turn_key
    assert all(run.turn_key != "same-caller-request-id" for run in captured_runs)

  _run(case())


def test_turn_key_and_registry_survive_stream_retry_event(tmp_path) -> None:
  async def case() -> None:
    captured: dict[str, Any] = {}

    class RetryRunner:
      def __init__(
        self,
        event_log: EventLog,
        ui_run,
        capability_execution: Any,
      ) -> None:
        self.event_log = event_log
        self.ui_run = ui_run
        self.capability_execution = capability_execution

      async def run(self, **_kwargs: Any) -> None:
        captured["before_retry"] = self.ui_run
        self.event_log.append({"type": "stream_retry", "attempt": 1, "error": "retry"})
        await asyncio.sleep(0)
        captured["after_retry"] = self.ui_run
        self.event_log.append({"type": "stream_complete", "usage": {}})

    async def build_chat_runtime(*, request, **_kwargs: Any) -> ChatRuntime:
      captured["request_run"] = request._ui_blocks_run
      capability_bind = request.capability_bind
      assert capability_bind is not None
      return ChatRuntime(
        system_prompt="test",
        build_runner=lambda event_log, _sid, _started_at: RetryRunner(
          event_log,
          request._ui_blocks_run,
          request.capability_execution,
        ),
        capability_execution=request.capability_execution,
      )

    _configure_runtime_builder(build_chat_runtime)
    await _dispatch_chat_turn(
      _session(),
      ChatTurnInputs(
        messages=[ChatMessage(role="user", content="show me")],
        request_id="request-id",
        context={},
        metadata={},
        model_key=None,
      ),
      event_log=EventLog(),
      on_event=_no_event,
      build_chat_runtime=build_chat_runtime,
      transcript_dir=tmp_path,
    )
    assert captured["before_retry"] is captured["request_run"]
    assert captured["after_retry"] is captured["request_run"]
    assert captured["after_retry"].registry is captured["before_retry"].registry
    assert captured["after_retry"].turn_key == captured["before_retry"].turn_key

  _run(case())


def test_ui_blocks_ready_dataclass_coercion_round_trip() -> None:
  raw = {
    "type": "ui_blocks_ready",
    "session_id": "sess-1",
    "skill_run_id": None,
    "turn_key": "turn-key",
    "emission_index": 3,
    "ui_blocks_id": "ub_deadbeefdeadbeef",
    "contract_version": 1,
    "payload": {"kind": "hank_ui_blocks.v1", "contract_version": 1, "blocks": []},
    "text_fallback": "Fallback",
    "ts": 123.5,
  }
  event = event_from_dict(raw)
  assert isinstance(event, UiBlocksReadyEvent)
  assert event_to_dict(event) == raw
