from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import timedelta
import json
import os
from pathlib import Path
import stat
from threading import Barrier
from typing import Any

from agent_workflow_contracts import models as workflow_models

import pytest

from agent_gateway.approval_policy import (
  ApprovalRequest,
  ApprovalVote,
  RunContext,
  build_approval_request,
  sha256_args,
  utc_now,
)
from agent_gateway.approval_store import (
  ApprovalExecutableBindingError,
  ExecutableApprovalBinding,
  PolicyGenerationMismatch,
  RootTurnRecordError,
  SQLiteApprovalStore,
  verify_policy_generation,
)
from agent_gateway.approvals import _approval_request_to_dict
from agent_gateway.tool_policy_registry import PreparedToolCall


def test_approval_store_creates_private_inode_and_sidecars(
  tmp_path: Path,
) -> None:
  path = tmp_path / "approvals.sqlite3"

  prior_umask = os.umask(0o022)
  try:
    store = SQLiteApprovalStore(path)
  finally:
    os.umask(prior_umask)

  file_stat = path.stat()
  assert stat.S_IMODE(file_stat.st_mode) == 0o600
  assert file_stat.st_uid == os.geteuid()
  assert file_stat.st_nlink == 1

  with store._connection():
    for suffix in ("-wal", "-shm"):
      sidecar_stat = Path(f"{path}{suffix}").stat()
      assert stat.S_IMODE(sidecar_stat.st_mode) == 0o600
      assert sidecar_stat.st_uid == os.geteuid()
      assert sidecar_stat.st_nlink == 1


def test_approval_store_rejects_bound_inode_replacement(
  tmp_path: Path,
) -> None:
  path = tmp_path / "approvals.sqlite3"
  store = SQLiteApprovalStore(path)
  original = path.stat()
  os.rename(path, tmp_path / "approvals.original.sqlite3")
  replacement_fd = os.open(
    path,
    os.O_RDWR | os.O_CREAT | os.O_EXCL,
    0o600,
  )
  os.close(replacement_fd)

  with pytest.raises(RuntimeError, match="identity"):
    store._connect()

  assert path.stat().st_ino != original.st_ino


def test_signed_approval_store_rejects_permission_drift(
  tmp_path: Path,
) -> None:
  path = tmp_path / "approvals.sqlite3"
  SQLiteApprovalStore(path)
  identity = path.stat()
  path.chmod(0o640)

  with pytest.raises(RuntimeError, match="permissions"):
    SQLiteApprovalStore(
      path,
      expected_device=identity.st_dev,
      expected_inode=identity.st_ino,
    )


def test_approval_store_rejects_writable_parent(
  tmp_path: Path,
) -> None:
  parent = tmp_path / "unsafe"
  parent.mkdir(mode=0o700)
  parent.chmod(0o777)

  with pytest.raises(RuntimeError, match="parent"):
    SQLiteApprovalStore(parent / "approvals.sqlite3")


def test_approval_store_rejects_missing_parent_without_creating_it(
  tmp_path: Path,
) -> None:
  parent = tmp_path / "missing"

  with pytest.raises(RuntimeError, match="parent is unavailable"):
    SQLiteApprovalStore(parent / "approvals.sqlite3")

  assert not parent.exists()


def test_approval_store_rejects_sidecar_symlink(
  tmp_path: Path,
) -> None:
  path = tmp_path / "approvals.sqlite3"
  target = tmp_path / "target"
  target.write_bytes(b"")
  Path(f"{path}-wal").symlink_to(target)

  with pytest.raises(RuntimeError, match="unavailable"):
    SQLiteApprovalStore(path)


def _binding_arguments(
  request: ApprovalRequest,
  prepared: PreparedToolCall,
) -> dict[str, Any]:
  return {
    "approval_id": request.approval_id,
    "job_id": "job-original",
    "tool_call_id": request.tool_call_id,
    "nonce": "nonce-original",
    "tenant_storage_id": "alice",
    "tool_name": request.tool_name,
    "prepared_call": prepared,
    "dispatch_context": {
      "skill_run_id": "skill-original",
      "step_id": "step-original",
      "workspace_dir": "/workspace/original",
      "batch_id": "batch-original",
      "call_index": 2,
      "capture_readable_resource_snapshot": True,
      "allow_uncertain_mcp_replay": False,
    },
    "credential_generation_ref": "grant-generation-original",
  }


async def _persist_binding(
  store: SQLiteApprovalStore,
) -> tuple[ApprovalRequest, PreparedToolCall]:
  prepared = PreparedToolCall(
    prepared_input={
      "token": "secret-original-operand",
      "orders": [{"ticker": "AAPL", "quantity": 3}],
    },
    exact_backend="remote",
  )
  request = build_approval_request(
    tool_call_id="call-original",
    tool_name="registered_write",
    tool_class="external_write",
    tool_args_redacted={
      "token": "[REDACTED]",
      "orders": [{"ticker": "AAPL", "quantity": 3}],
    },
    args_hash=sha256_args(prepared.materialize_input()),
    run_context=RunContext(
      user_id="alice",
      request_id="request-original",
      session_id="session-original",
      run_id="run-original",
      tenant_id="tenant-public-id",
    ),
    state="pending_user",
  )
  await store.create(request)
  await store.record_executable_binding(**_binding_arguments(request, prepared))
  return request, prepared


def test_private_binding_cold_reopen_preserves_secret_and_current_decision(
  tmp_path: Path,
) -> None:
  async def case() -> None:
    path = tmp_path / "approvals.sqlite3"
    request, prepared = await _persist_binding(SQLiteApprovalStore(path))
    reopened = SQLiteApprovalStore(path)
    binding, pending = await reopened._read_executable_binding_for_executor(
      job_id="job-original",
      tool_call_id=request.tool_call_id,
      nonce="nonce-original",
      tenant_storage_id="alice",
    )
    assert pending.state == "pending_user"
    assert pending.decision is None
    await reopened.record_vote(
      request.approval_id,
      ApprovalVote(
        vote_id="vote-original",
        approval_id=request.approval_id,
        decider_id="alice",
        decider_role=None,
        decision="approved",
        decision_reason=None,
        decided_at=utc_now(),
      ),
    )
    binding, approved = await SQLiteApprovalStore(
      path,
    )._read_executable_binding_for_executor(
      job_id="job-original",
      tool_call_id=request.tool_call_id,
      nonce="nonce-original",
      tenant_storage_id="alice",
    )
    executable = PreparedToolCall(
      prepared_input=binding.prepared_input,
      exact_backend=binding.exact_backend,
    )
    assert executable.materialize_input() == prepared.materialize_input()
    assert executable.exact_backend == "remote"
    assert binding.credential_generation_ref == "grant-generation-original"
    assert binding.dispatch_context == _binding_arguments(
      request, prepared,
    )["dispatch_context"]
    assert approved.approval_id == request.approval_id
    assert approved.state == "approved"
    assert approved.decision == "approved"
    assert approved.state_version == pending.state_version + 1
    with pytest.raises(TypeError):
      binding.prepared_input["token"] = "replacement"  # type: ignore[index]
    with pytest.raises(TypeError):
      binding.dispatch_context["step_id"] = "replacement"  # type: ignore[index]
    detached = executable.materialize_input()
    detached["token"] = "replacement"
    assert executable.materialize_input()["token"] == "secret-original-operand"

    for public in (
      await reopened.get(request.approval_id),
      await reopened.get_by_tool_call_id(request.tool_call_id),
    ):
      assert public is not None
      projection = _approval_request_to_dict(public)
      assert projection["tool_args_redacted"]["token"] == "[REDACTED]"
      serialized = json.dumps(projection)
      assert "secret-original-operand" not in serialized
      assert "grant-generation-original" not in serialized
      assert "/workspace/original" not in serialized
    assert "secret-original-operand" not in repr(binding)
    assert "secret-original-operand" not in str(binding)

  asyncio.run(case())


@pytest.mark.parametrize(
  ("scope", "code"),
  [
    ({"nonce": "nonce-other"}, "nonce_mismatch"),
    ({"tenant_storage_id": "bob"}, "tenant_mismatch"),
    ({"job_id": "job-other"}, "missing_binding"),
    ({"tool_call_id": "call-other"}, "missing_binding"),
  ],
)
def test_private_binding_checks_scope_before_decoding_operands(
  tmp_path: Path,
  scope: dict[str, str],
  code: str,
) -> None:
  async def case() -> None:
    store = SQLiteApprovalStore(tmp_path / "approvals.sqlite3")
    request, _ = await _persist_binding(store)
    with store._connection() as conn:
      conn.execute(
        """
        UPDATE approval_executable_bindings SET prepared_input_json = ?
        WHERE approval_id = ?
        """,
        ("invalid JSON secret-original-operand", request.approval_id),
      )
    lookup = {
      "job_id": "job-original",
      "tool_call_id": request.tool_call_id,
      "nonce": "nonce-original",
      "tenant_storage_id": "alice",
      **scope,
    }
    with pytest.raises(ApprovalExecutableBindingError) as error:
      await store._read_executable_binding_for_executor(**lookup)
    assert error.value.code == code
    assert "secret-original-operand" not in str(error.value)
    assert "secret-original-operand" not in repr(error.value)

  asyncio.run(case())


@pytest.mark.parametrize(
  ("field", "replacement", "code"),
  [
    ("nonce", "nonce-other", "nonce_mismatch"),
    ("tenant_storage_id", "bob", "tenant_mismatch"),
    ("job_id", "job-other", "call_identity_mismatch"),
    ("tool_call_id", "call-other", "call_identity_mismatch"),
    (
      "prepared_call",
      PreparedToolCall({"token": "different-secret"}, exact_backend="remote"),
      "operand_mismatch",
    ),
    (
      "prepared_call",
      PreparedToolCall(
        {"token": "secret-original-operand", "orders": [{"ticker": "AAPL", "quantity": 3}]},
        exact_backend="local",
      ),
      "operand_mismatch",
    ),
    ("credential_generation_ref", "grant-generation-other", "operand_mismatch"),
    ("dispatch_context", {"step_id": "step-other"}, "operand_mismatch"),
  ],
)
def test_private_binding_contradiction_does_not_replace_original(
  tmp_path: Path,
  field: str,
  replacement: object,
  code: str,
) -> None:
  async def case() -> None:
    store = SQLiteApprovalStore(tmp_path / "approvals.sqlite3")
    request, prepared = await _persist_binding(store)
    arguments = _binding_arguments(request, prepared)
    await store.record_executable_binding(**arguments)
    with pytest.raises(ApprovalExecutableBindingError) as error:
      await store.record_executable_binding(**{**arguments, field: replacement})
    assert error.value.code == code
    binding, current = await store._read_executable_binding_for_executor(
      job_id="job-original",
      tool_call_id=request.tool_call_id,
      nonce="nonce-original",
      tenant_storage_id="alice",
    )
    assert PreparedToolCall(binding.prepared_input).materialize_input() == prepared.materialize_input()
    assert binding.exact_backend == "remote"
    assert binding.credential_generation_ref == "grant-generation-original"
    assert current.state == "pending_user"
    assert "secret-original-operand" not in str(error.value)
    assert "different-secret" not in str(error.value)

  asyncio.run(case())


@pytest.mark.parametrize(
  ("column", "replacement"),
  [
    ("prepared_input_json", '{"token":"different-secret"}'),
    ("dispatch_context_json", '{"step_id":"step-other"}'),
    ("nonce", "nonce-corrupted"),
  ],
)
def test_private_binding_detects_operand_or_identity_corruption(
  tmp_path: Path,
  column: str,
  replacement: str,
) -> None:
  async def case() -> None:
    store = SQLiteApprovalStore(tmp_path / "approvals.sqlite3")
    request, prepared = await _persist_binding(store)
    with store._connection() as conn:
      conn.execute(
        f"UPDATE approval_executable_bindings SET {column} = ? WHERE approval_id = ?",
        (replacement, request.approval_id),
      )
    with pytest.raises(ApprovalExecutableBindingError) as error:
      await store._read_executable_binding_for_executor(
        job_id="job-original",
        tool_call_id=request.tool_call_id,
        nonce="nonce-corrupted" if column == "nonce" else "nonce-original",
        tenant_storage_id="alice",
      )
    assert error.value.code == "operand_mismatch"
    assert "different-secret" not in str(error.value)
    assert "secret-original-operand" not in str(error.value)
    with pytest.raises(ApprovalExecutableBindingError) as replay:
      await store.record_executable_binding(**_binding_arguments(request, prepared))
    assert replay.value.code == (
      "nonce_mismatch" if column == "nonce" else "operand_mismatch"
    )

  asyncio.run(case())


def test_bound_approval_arguments_cannot_change_after_binding(
  tmp_path: Path,
) -> None:
  async def case() -> None:
    store = SQLiteApprovalStore(tmp_path / "approvals.sqlite3")
    request, _ = await _persist_binding(store)
    with pytest.raises(ApprovalExecutableBindingError) as error:
      await store.update_request(
        replace(
          request,
          args_hash="changed-operand-hash",
          tool_args_redacted={"token": "[CHANGED]"},
        ),
      )
    assert error.value.code == "operand_mismatch"
    public = await store.get(request.approval_id)
    assert public is not None
    assert public.args_hash == request.args_hash
    assert public.tool_args_redacted == request.tool_args_redacted

  asyncio.run(case())


def test_first_private_binding_cannot_reassign_approval_tenant(
  tmp_path: Path,
) -> None:
  async def case() -> None:
    original = SQLiteApprovalStore(tmp_path / "original.sqlite3")
    request, prepared = await _persist_binding(original)
    store = SQLiteApprovalStore(tmp_path / "reopened-copy.sqlite3")
    await store.create(request)
    arguments = _binding_arguments(request, prepared)
    with pytest.raises(ApprovalExecutableBindingError) as error:
      await store.record_executable_binding(
        **{**arguments, "tenant_storage_id": "bob"},
      )
    assert error.value.code == "tenant_mismatch"
    with pytest.raises(ApprovalExecutableBindingError) as missing:
      await store._read_executable_binding_for_executor(
        job_id="job-original",
        tool_call_id=request.tool_call_id,
        nonce="nonce-original",
        tenant_storage_id="bob",
      )
    assert missing.value.code == "missing_binding"
    await store.record_executable_binding(**arguments)
    binding, _ = await store._read_executable_binding_for_executor(
      job_id="job-original",
      tool_call_id=request.tool_call_id,
      nonce="nonce-original",
      tenant_storage_id="alice",
    )
    assert binding.prepared_input["token"] == "secret-original-operand"

  asyncio.run(case())


def test_private_binding_cannot_replace_original_approval_identity(
  tmp_path: Path,
) -> None:
  async def case() -> None:
    store = SQLiteApprovalStore(tmp_path / "approvals.sqlite3")
    request, prepared = await _persist_binding(store)
    other = replace(
      request,
      approval_id="approval-other",
      approval_chain_id="approval-other",
    )
    await store.create(other)
    with pytest.raises(ApprovalExecutableBindingError) as error:
      await store.record_executable_binding(**_binding_arguments(other, prepared))
    assert error.value.code == "call_identity_mismatch"
    binding, decision = await store._read_executable_binding_for_executor(
      job_id="job-original",
      tool_call_id=request.tool_call_id,
      nonce="nonce-original",
      tenant_storage_id="alice",
    )
    assert binding.approval_id == request.approval_id
    assert decision.approval_id == request.approval_id
    assert binding.prepared_input["token"] == "secret-original-operand"

  asyncio.run(case())


async def _approved_executable_binding(
  store: SQLiteApprovalStore,
) -> tuple[ExecutableApprovalBinding, ApprovalRequest]:
  request, _ = await _persist_binding(store)
  await store.record_vote(
    request.approval_id,
    ApprovalVote(
      vote_id="vote-original",
      approval_id=request.approval_id,
      decider_id="alice",
      decider_role=None,
      decision="approved",
      decision_reason=None,
      decided_at=utc_now(),
    ),
  )
  return await store._read_executable_binding_for_executor(
    job_id="job-original",
    tool_call_id=request.tool_call_id,
    nonce="nonce-original",
    tenant_storage_id="alice",
  )


def test_original_call_claim_is_atomic_across_independent_store_connections(
  tmp_path: Path,
) -> None:
  path = tmp_path / "approvals.sqlite3"
  first = SQLiteApprovalStore(path)
  binding, decision = asyncio.run(_approved_executable_binding(first))
  second = SQLiteApprovalStore(path)
  barrier = Barrier(2)

  def claim(store: SQLiteApprovalStore, token: str) -> str:
    barrier.wait()
    try:
      result = asyncio.run(store._claim_executable_binding_for_executor(
        binding=binding,
        decision_state_version=decision.state_version,
        claim_token=token,
      ))
    except ApprovalExecutableBindingError as error:
      return error.code
    assert result is None
    return token

  with ThreadPoolExecutor(max_workers=2) as pool:
    futures = [
      pool.submit(claim, first, "token-first"),
      pool.submit(claim, second, "token-second"),
    ]
    outcomes = [future.result() for future in futures]
  winners = [
    outcome for outcome in outcomes
    if outcome in {"token-first", "token-second"}
  ]
  assert len(winners) == 1
  assert outcomes.count("execution_outcome_unknown") == 1
  known_result = ({"committed": True, "secret": "secret-returned-result"}, None)
  asyncio.run(first._complete_executable_binding_for_executor(
    binding=binding,
    decision_state_version=decision.state_version,
    claim_token=winners[0],
    tool_result=known_result,
  ))
  replay = asyncio.run(SQLiteApprovalStore(path)._claim_executable_binding_for_executor(
    binding=binding,
    decision_state_version=decision.state_version,
    claim_token="token-later",
  ))
  assert replay == known_result


def test_claim_without_returned_result_stays_unknown_after_cold_reopen(
  tmp_path: Path,
) -> None:
  async def case() -> None:
    path = tmp_path / "approvals.sqlite3"
    store = SQLiteApprovalStore(path)
    binding, decision = await _approved_executable_binding(store)
    assert await store._claim_executable_binding_for_executor(
      binding=binding,
      decision_state_version=decision.state_version,
      claim_token="token-original",
    ) is None
    reopened = SQLiteApprovalStore(path)
    for token in ("token-original", "token-after-process-loss"):
      with pytest.raises(ApprovalExecutableBindingError) as error:
        await reopened._claim_executable_binding_for_executor(
          binding=binding,
          decision_state_version=decision.state_version,
          claim_token=token,
        )
      assert error.value.code == "execution_outcome_unknown"
    public = await reopened.get(binding.approval_id)
    assert public is not None
    assert public.state == "approved"
    assert public.state_version == decision.state_version

  asyncio.run(case())


@pytest.mark.parametrize(
  "tool_result",
  [
    ({"secret": "secret-returned-result", "rows": [1, 2]}, None),
    (
      None,
      {
        "code": "trade_preview_stale",
        "next_call": {"tool": "preview_trade", "arguments": {"ticker": "AAPL"}},
      },
    ),
    (None, None),
  ],
)
def test_completed_original_attempt_cold_replays_exact_private_tool_result(
  tmp_path: Path,
  tool_result: tuple[Any, dict[str, Any] | None],
) -> None:
  async def case() -> None:
    path = tmp_path / "approvals.sqlite3"
    store = SQLiteApprovalStore(path)
    binding, decision = await _approved_executable_binding(store)
    assert await store._read_completed_executable_result_for_executor(
      binding=binding,
    ) is None
    assert await store._read_completed_executable_result_for_executor(
      binding=binding,
    ) is None
    assert await store._claim_executable_binding_for_executor(
      binding=binding,
      decision_state_version=decision.state_version,
      claim_token="token-original",
    ) is None
    await store._complete_executable_binding_for_executor(
      binding=binding,
      decision_state_version=decision.state_version,
      claim_token="token-original",
      tool_result=tool_result,
    )
    reopened = SQLiteApprovalStore(path)
    assert await reopened._read_completed_executable_result_for_executor(
      binding=binding,
    ) == tool_result
    assert await reopened._claim_executable_binding_for_executor(
      binding=binding,
      decision_state_version=decision.state_version,
      claim_token="token-next",
    ) == tool_result
    await reopened._complete_executable_binding_for_executor(
      binding=binding,
      decision_state_version=decision.state_version,
      claim_token="token-original",
      tool_result=tool_result,
    )
    public = await reopened.get(binding.approval_id)
    assert public is not None
    assert public.state == "approved"
    assert public.state_version == decision.state_version
    projection = _approval_request_to_dict(public)
    assert "secret-returned-result" not in json.dumps(projection)
    assert "execution_result_json" not in projection
    assert "execution_claim_token" not in projection

  asyncio.run(case())


def test_pending_and_stale_decision_claims_do_not_burn_original_attempt(
  tmp_path: Path,
) -> None:
  async def case() -> None:
    store = SQLiteApprovalStore(tmp_path / "approvals.sqlite3")
    request, _ = await _persist_binding(store)
    binding, pending = await store._read_executable_binding_for_executor(
      job_id="job-original",
      tool_call_id=request.tool_call_id,
      nonce="nonce-original",
      tenant_storage_id="alice",
    )
    with pytest.raises(ApprovalExecutableBindingError) as pending_error:
      await store._claim_executable_binding_for_executor(
        binding=binding,
        decision_state_version=pending.state_version,
        claim_token="token-pending",
      )
    assert pending_error.value.code == "decision_mismatch"
    approved = await store.transition_state(
      request.approval_id, "approved",
      expected_state_version=pending.state_version,
      decider_id="alice",
    )
    with pytest.raises(ApprovalExecutableBindingError) as stale_error:
      await store._claim_executable_binding_for_executor(
        binding=binding,
        decision_state_version=pending.state_version,
        claim_token="token-stale-decision",
      )
    assert stale_error.value.code == "decision_mismatch"
    assert await store._claim_executable_binding_for_executor(
      binding=binding,
      decision_state_version=approved.state_version,
      claim_token="token-current-decision",
    ) is None

  asyncio.run(case())


@pytest.mark.parametrize(
  ("alteration", "code"),
  [
    ({"nonce": "nonce-other"}, "nonce_mismatch"),
    ({"tenant_storage_id": "bob"}, "tenant_mismatch"),
    ({"credential_generation_ref": "grant-other"}, "operand_mismatch"),
    ({"operand_digest": "sha256:corrupted"}, "operand_mismatch"),
  ],
)
def test_claim_reverifies_original_binding_before_reserving_call(
  tmp_path: Path,
  alteration: dict[str, Any],
  code: str,
) -> None:
  async def case() -> None:
    store = SQLiteApprovalStore(tmp_path / "approvals.sqlite3")
    binding, decision = await _approved_executable_binding(store)
    with pytest.raises(ApprovalExecutableBindingError) as error:
      await store._claim_executable_binding_for_executor(
        binding=replace(binding, **alteration),
        decision_state_version=decision.state_version,
        claim_token="token-altered",
      )
    assert error.value.code == code
    assert await store._claim_executable_binding_for_executor(
      binding=binding,
      decision_state_version=decision.state_version,
      claim_token="token-original",
    ) is None

  asyncio.run(case())


def test_completion_requires_original_token_and_cannot_replace_returned_result(
  tmp_path: Path,
) -> None:
  async def case() -> None:
    store = SQLiteApprovalStore(tmp_path / "approvals.sqlite3")
    binding, decision = await _approved_executable_binding(store)
    assert await store._claim_executable_binding_for_executor(
      binding=binding,
      decision_state_version=decision.state_version,
      claim_token="token-original",
    ) is None
    result = ({"secret": "secret-returned-result"}, None)
    with pytest.raises(ApprovalExecutableBindingError) as wrong_token:
      await store._complete_executable_binding_for_executor(
        binding=binding,
        decision_state_version=decision.state_version,
        claim_token="token-other",
        tool_result=result,
      )
    assert wrong_token.value.code == "claim_token_mismatch"
    await store._complete_executable_binding_for_executor(
      binding=binding,
      decision_state_version=decision.state_version,
      claim_token="token-original",
      tool_result=result,
    )
    with pytest.raises(ApprovalExecutableBindingError) as contradiction:
      await store._complete_executable_binding_for_executor(
        binding=binding,
        decision_state_version=decision.state_version,
        claim_token="token-original",
        tool_result=({"secret": "replacement-result"}, None),
      )
    assert contradiction.value.code == "result_mismatch"
    assert await store._claim_executable_binding_for_executor(
      binding=binding,
      decision_state_version=decision.state_version,
      claim_token="token-next",
    ) == result

  asyncio.run(case())


def test_completed_result_corruption_never_replays_or_exposes_private_result(
  tmp_path: Path,
) -> None:
  async def case() -> None:
    store = SQLiteApprovalStore(tmp_path / "approvals.sqlite3")
    binding, decision = await _approved_executable_binding(store)
    assert await store._claim_executable_binding_for_executor(
      binding=binding,
      decision_state_version=decision.state_version,
      claim_token="token-original",
    ) is None
    await store._complete_executable_binding_for_executor(
      binding=binding,
      decision_state_version=decision.state_version,
      claim_token="token-original",
      tool_result=({"secret": "secret-returned-result"}, None),
    )
    with store._connection() as conn:
      conn.execute(
        """
        UPDATE approval_executable_bindings SET execution_result_json = ?
        WHERE approval_id = ?
        """,
        ('[{"secret":"corrupted-secret"},null]', binding.approval_id),
      )
    with pytest.raises(ApprovalExecutableBindingError) as error:
      await store._claim_executable_binding_for_executor(
        binding=binding,
        decision_state_version=decision.state_version,
        claim_token="token-next",
      )
    assert error.value.code == "result_mismatch"
    assert "secret-returned-result" not in str(error.value)
    assert "corrupted-secret" not in str(error.value)

  asyncio.run(case())


@pytest.mark.parametrize("as_completion", [False, True])
def test_completed_attempt_preserves_canonical_result_models_and_child_evidence(
  tmp_path: Path,
  as_completion: bool,
) -> None:
  async def case() -> None:
    digest = "sha256:" + "a" * 64
    result = workflow_models.TaskResult(
      task_result_id="result-original",
      logical_task=workflow_models.OrdinaryDelegationTaskRef(
        delegation_id="delegation-original",
        operation=workflow_models.AgentOperationRef(
          namespace="research",
          name="original",
          version="1.0",
          digest=digest,
        ),
      ),
      attempt=workflow_models.AttemptRef(
        attempt_number=1,
        attempt_id="attempt-original",
        physical_task_id="child-original",
      ),
      execution=workflow_models.ExecutionSettlement(
        status="failed", terminal_reason="runtime_error",
      ),
      observation=workflow_models.TaskObservation(
        transcript=workflow_models.TranscriptHandle(
          kind="child_transcript", owner_id="child-original",
        ),
        activity=workflow_models.ActivityHandle(
          kind="child_activity", owner_id="child-original",
        ),
      ),
      provenance=workflow_models.TaskResultProvenance(
        admitted_task_digest=digest,
        model_bind_digest=digest,
        capability_binding_digest=digest,
        tool_grant_digest=digest,
      ),
    )
    value = (
      workflow_models.AgentCompletionEnvelope(
        message_id="message-original",
        task_result_ref=workflow_models.TaskResultRef.from_result(result),
        settlement_projection=workflow_models.SettlementProjection(
          execution_status="failed", terminal_reason="runtime_error",
        ),
        parent_materialization=None,
        child_evidence=workflow_models.ChildEvidenceProjection(
          evidence_tools=("memory_write",),
        ),
        stage_refs={"private_context": "secret-returned-result"},
      )
      if as_completion else result
    )
    path = tmp_path / "approvals.sqlite3"
    store = SQLiteApprovalStore(path)
    binding, decision = await _approved_executable_binding(store)
    assert await store._claim_executable_binding_for_executor(
      binding=binding,
      decision_state_version=decision.state_version,
      claim_token="token-original",
    ) is None
    await store._complete_executable_binding_for_executor(
      binding=binding,
      decision_state_version=decision.state_version,
      claim_token="token-original",
      tool_result=(value, None),
    )
    reopened = SQLiteApprovalStore(path)
    replay = await reopened._claim_executable_binding_for_executor(
      binding=binding,
      decision_state_version=decision.state_version,
      claim_token="token-next",
    )
    assert replay is not None
    assert replay[1] is None
    assert type(replay[0]) is type(value)
    assert replay[0] == value
    if as_completion:
      assert isinstance(replay[0], workflow_models.AgentCompletionEnvelope)
      assert replay[0].child_evidence is not None
      assert replay[0].child_evidence.evidence_tools == ("memory_write",)
      assert replay[0].stage_refs["private_context"] == "secret-returned-result"
    public = await reopened.get(binding.approval_id)
    assert public is not None
    assert "secret-returned-result" not in json.dumps(_approval_request_to_dict(public))

  asyncio.run(case())


def test_claimed_result_completes_after_expiry_clock_without_changing_approval(
  tmp_path: Path,
) -> None:
  async def case() -> None:
    path = tmp_path / "approvals.sqlite3"
    store = SQLiteApprovalStore(path)
    request, _ = await _persist_binding(store)
    expires_at = request.requested_at + timedelta(minutes=20)
    await store.transition_state(
      request.approval_id, "pending_user",
      expires_at=expires_at,
      expected_state_version=request.state_version,
    )
    await store.record_vote(
      request.approval_id,
      ApprovalVote(
        vote_id="vote-original",
        approval_id=request.approval_id,
        decider_id="alice",
        decider_role=None,
        decision="approved",
        decision_reason=None,
        decided_at=utc_now(),
      ),
    )
    binding, decision = await store._read_executable_binding_for_executor(
      job_id="job-original",
      tool_call_id=request.tool_call_id,
      nonce="nonce-original",
      tenant_storage_id="alice",
    )
    assert decision.expires_at == expires_at
    assert await store._claim_executable_binding_for_executor(
      binding=binding,
      decision_state_version=decision.state_version,
      claim_token="token-original",
    ) is None
    # Exercise the existing expiry owner at a later clock, rather than
    # fabricating an expired/denied approval that it would not produce.
    maintenance = await store.maintain_pending(
      now=expires_at + timedelta(seconds=1),
    )
    assert maintenance.approvals_expired == 0
    result = ({"committed": True, "secret": "secret-returned-result"}, None)
    await store._complete_executable_binding_for_executor(
      binding=binding,
      decision_state_version=decision.state_version,
      claim_token="token-original",
      tool_result=result,
    )
    reopened = SQLiteApprovalStore(path)
    assert await reopened._read_completed_executable_result_for_executor(
      binding=binding,
    ) == result
    assert await reopened._claim_executable_binding_for_executor(
      binding=binding,
      decision_state_version=decision.state_version,
      claim_token="token-next",
    ) == result
    public = await reopened.get(binding.approval_id)
    assert public is not None
    assert _approval_request_to_dict(public) == _approval_request_to_dict(decision)

  asyncio.run(case())


def test_claimed_outcome_settlement_and_readback_do_not_reauthorize_invocation(
  tmp_path: Path,
) -> None:
  async def case() -> None:
    path = tmp_path / "approvals.sqlite3"
    store = SQLiteApprovalStore(path)
    binding, original_decision = await _approved_executable_binding(store)
    assert await store._claim_executable_binding_for_executor(
      binding=binding,
      decision_state_version=original_decision.state_version,
      claim_token="token-original",
    ) is None
    # This is a real later owner decision, not a clock-manufactured denial.
    later_decision = await store.transition_state(
      binding.approval_id, "denied",
      expected_state_version=original_decision.state_version,
      decider_id="alice",
      decision_reason="Owner changed authorization after invocation began",
    )
    result = ({"committed": True, "secret": "secret-returned-result"}, None)
    with pytest.raises(ApprovalExecutableBindingError) as wrong_version:
      await store._complete_executable_binding_for_executor(
        binding=binding,
        decision_state_version=later_decision.state_version,
        claim_token="token-original",
        tool_result=result,
      )
    assert wrong_version.value.code == "decision_mismatch"
    await store._complete_executable_binding_for_executor(
      binding=binding,
      decision_state_version=original_decision.state_version,
      claim_token="token-original",
      tool_result=result,
    )
    reopened = SQLiteApprovalStore(path)
    assert await reopened._read_completed_executable_result_for_executor(
      binding=binding,
    ) == result
    assert await reopened._claim_executable_binding_for_executor(
      binding=binding,
      decision_state_version=later_decision.state_version,
      claim_token="token-readback",
    ) == result
    public = await reopened.get(binding.approval_id)
    assert public is not None
    assert public.state == "denied"
    assert public.state_version == later_decision.state_version
    assert _approval_request_to_dict(public) == _approval_request_to_dict(later_decision)

  asyncio.run(case())


async def _root_turn_arguments(
  store: SQLiteApprovalStore,
  *,
  turn_id: str = "turn-original",
) -> dict[str, Any]:
  from agent_gateway.single_user_policy import SingleUserApprovalPolicy

  policy = SingleUserApprovalPolicy()
  generation = await store.record_policy_generation(
    policy_id=policy.policy_id,
    policy_version=policy.policy_version,
    policy_bundle_hash=policy.policy_bundle_hash,
  )
  return {
    "job_id": "job-original",
    "turn_id": turn_id,
    "tenant_storage_id": "alice",
    "capability_bind": {
      "model_key": "session.driver",
      "provider": "anthropic",
      "credential_ref": "grant-generation-original",
    },
    "credential_generation_ref": "grant-generation-original",
    "effective_grant": {"tool_ids": ["registered_write"], "withheld_tools": []},
    "routes": {"registered_write": {"backend": "remote", "server": "registered"}},
    "research_binding": {"file_id": 12, "revision": "research-original"},
    "selection_snapshot": {"role": "analyst", "effort": "high"},
    "approval_route_kind": "DurableLocalApprovalRoute",
    "policy_generation_ref": generation.generation_ref,
  }


def test_root_turn_cold_reopen_preserves_exact_private_authority(
  tmp_path: Path,
) -> None:
  async def case() -> None:
    path = tmp_path / "approvals.sqlite3"
    store = SQLiteApprovalStore(path)
    arguments = await _root_turn_arguments(store)
    await store.record_root_turn(**arguments)
    await store.record_root_turn(**{**arguments, "turn_id": "turn-second"})
    reopened = SQLiteApprovalStore(path)
    record = await reopened._read_root_turn_for_executor(
      job_id="job-original", turn_id="turn-original", tenant_storage_id="alice",
    )
    assert record is not None
    assert record.job_id == "job-original"
    assert record.turn_id == "turn-original"
    assert record.tenant_storage_id == "alice"
    assert json.loads(record.capability_bind_json) == arguments["capability_bind"]
    assert record.credential_generation_ref == arguments["credential_generation_ref"]
    assert json.loads(record.effective_grant_json) == arguments["effective_grant"]
    assert json.loads(record.routes_json) == arguments["routes"]
    assert json.loads(record.research_binding_json) == arguments["research_binding"]
    assert json.loads(record.selection_snapshot_json) == arguments["selection_snapshot"]
    assert record.approval_route_kind == arguments["approval_route_kind"]
    assert record.policy_generation_ref == arguments["policy_generation_ref"]
    assert record.created_at.tzinfo is not None
    assert record.executable_refs == ()
    second = await reopened._read_root_turn_for_executor(
      job_id="job-original", turn_id="turn-second", tenant_storage_id="alice",
    )
    assert second is not None
    assert second.turn_id == "turn-second"
    assert second.capability_bind_json == record.capability_bind_json
    assert await reopened._read_root_turn_for_executor(
      job_id="job-original", turn_id="turn-original", tenant_storage_id="bob",
    ) is None
    assert await reopened._read_root_turn_for_executor(
      job_id="job-original", turn_id="absent", tenant_storage_id="alice",
    ) is None

  asyncio.run(case())


def test_root_turn_replay_preserves_creation_time_and_attached_call(
  tmp_path: Path,
) -> None:
  async def case() -> None:
    path = tmp_path / "approvals.sqlite3"
    store = SQLiteApprovalStore(path)
    arguments = await _root_turn_arguments(store)
    await store.record_root_turn(**arguments)
    request, _ = await _persist_binding(store)
    await store.link_root_turn_executable_binding(
      job_id="job-original", turn_id="turn-original",
      tool_call_id=request.tool_call_id, nonce="nonce-original",
      tenant_storage_id="alice",
    )
    before = await store._read_root_turn_for_executor(
      job_id="job-original", turn_id="turn-original", tenant_storage_id="alice",
    )
    assert before is not None
    reopened = SQLiteApprovalStore(path)
    await reopened.record_root_turn(**arguments)
    await reopened.record_root_turn(**{**arguments, "created_at": before.created_at})
    after = await reopened._read_root_turn_for_executor(
      job_id="job-original", turn_id="turn-original", tenant_storage_id="alice",
    )
    assert after == before
    assert after is not None
    assert [(ref.tool_call_id, ref.nonce) for ref in after.executable_refs] == [
      (request.tool_call_id, "nonce-original"),
    ]

  asyncio.run(case())


@pytest.mark.parametrize(
  ("field", "replacement", "code"),
  [
    ("tenant_storage_id", "bob", "tenant_mismatch"),
    ("capability_bind", {"credential_ref": "replacement"}, "snapshot_mismatch"),
    ("credential_generation_ref", "replacement", "snapshot_mismatch"),
    ("effective_grant", {"tool_ids": ["another_tool"]}, "snapshot_mismatch"),
    ("routes", {"registered_write": {"backend": "local"}}, "snapshot_mismatch"),
    ("research_binding", None, "snapshot_mismatch"),
    ("selection_snapshot", {"effort": "low"}, "snapshot_mismatch"),
    ("approval_route_kind", "ControlPlaneApprovalRoute", "snapshot_mismatch"),
    ("policy_generation_ref", "replacement", "snapshot_mismatch"),
  ],
)
def test_root_turn_cold_replay_cannot_replace_original_authority(
  tmp_path: Path,
  field: str,
  replacement: object,
  code: str,
) -> None:
  async def case() -> None:
    path = tmp_path / "approvals.sqlite3"
    store = SQLiteApprovalStore(path)
    arguments = await _root_turn_arguments(store)
    await store.record_root_turn(**arguments)
    before = await store._read_root_turn_for_executor(
      job_id="job-original", turn_id="turn-original", tenant_storage_id="alice",
    )
    reopened = SQLiteApprovalStore(path)
    with pytest.raises(RootTurnRecordError) as error:
      await reopened.record_root_turn(**{**arguments, field: replacement})
    assert error.value.code == code
    assert await reopened._read_root_turn_for_executor(
      job_id="job-original", turn_id="turn-original", tenant_storage_id="alice",
    ) == before

  asyncio.run(case())


def test_root_turn_approval_locator_derives_exact_cold_binding_and_turn(
  tmp_path: Path,
) -> None:
  async def case() -> None:
    path = tmp_path / "approvals.sqlite3"
    store = SQLiteApprovalStore(path)
    arguments = await _root_turn_arguments(store)
    await store.record_root_turn(**arguments)
    await store.record_root_turn(**{**arguments, "turn_id": "turn-second"})
    request, _ = await _persist_binding(store)
    assert await store._locate_root_turn_for_approval(
      approval_id=request.approval_id, tenant_storage_id="alice",
    ) is None
    await store.link_root_turn_executable_binding(
      job_id="job-original", turn_id="turn-original",
      tool_call_id=request.tool_call_id, nonce="nonce-original",
      tenant_storage_id="alice",
    )
    reopened = SQLiteApprovalStore(path)
    await reopened.link_root_turn_executable_binding(
      job_id="job-original", turn_id="turn-original",
      tool_call_id=request.tool_call_id, nonce="nonce-original",
      tenant_storage_id="alice",
    )
    assert await reopened._locate_root_turn_for_approval(
      approval_id=request.approval_id, tenant_storage_id="alice",
    ) == ("job-original", request.tool_call_id, "nonce-original", "turn-original")
    assert await reopened._locate_root_turn_for_approval(
      approval_id=request.approval_id, tenant_storage_id="bob",
    ) is None
    assert await reopened._locate_root_turn_for_approval(
      approval_id="absent", tenant_storage_id="alice",
    ) is None
    with pytest.raises(RootTurnRecordError) as conflict:
      await reopened.link_root_turn_executable_binding(
        job_id="job-original", turn_id="turn-second",
        tool_call_id=request.tool_call_id, nonce="nonce-original",
        tenant_storage_id="alice",
      )
    assert conflict.value.code == "turn_link_mismatch"
    public = await reopened.get(request.approval_id)
    assert public is not None
    assert _approval_request_to_dict(public) == _approval_request_to_dict(request)
    assert "grant-generation-original" not in json.dumps(_approval_request_to_dict(public))

  asyncio.run(case())


@pytest.mark.parametrize(
  ("tenant", "nonce", "code"),
  [
    ("bob", "nonce-original", "tenant_mismatch"),
    ("alice", "wrong-nonce", "nonce_mismatch"),
  ],
)
def test_root_turn_link_cannot_adopt_other_tenant_or_nonce(
  tmp_path: Path,
  tenant: str,
  nonce: str,
  code: str,
) -> None:
  async def case() -> None:
    store = SQLiteApprovalStore(tmp_path / "approvals.sqlite3")
    await store.record_root_turn(**await _root_turn_arguments(store))
    request, _ = await _persist_binding(store)
    with pytest.raises((RootTurnRecordError, ApprovalExecutableBindingError)) as error:
      await store.link_root_turn_executable_binding(
        job_id="job-original", turn_id="turn-original",
        tool_call_id=request.tool_call_id, nonce=nonce, tenant_storage_id=tenant,
      )
    assert error.value.code == code
    assert await store._locate_root_turn_for_approval(
      approval_id=request.approval_id, tenant_storage_id="alice",
    ) is None

  asyncio.run(case())


@pytest.mark.parametrize(
  ("field", "replacement"),
  [
    ("policy_id", "another-policy"),
    ("policy_version", "next-version"),
    ("policy_bundle_hash", "b" * 64),
  ],
)
def test_policy_generation_cold_reverify_reports_typed_mismatch(
  tmp_path: Path,
  field: str,
  replacement: str,
) -> None:
  async def case() -> None:
    path = tmp_path / "approvals.sqlite3"
    store = SQLiteApprovalStore(path)
    arguments = await _root_turn_arguments(store)
    await store.record_root_turn(**arguments)
    original = await store.current_policy_generation()
    assert original is not None
    identity = {
      "policy_id": original.policy_id,
      "policy_version": original.policy_version,
      "policy_bundle_hash": original.policy_bundle_hash,
    }
    changed_identity = {**identity, field: replacement}
    changed = await store.record_policy_generation(
      policy_id=changed_identity["policy_id"],
      policy_version=changed_identity["policy_version"],
      policy_bundle_hash=changed_identity["policy_bundle_hash"],
    )
    reopened = SQLiteApprovalStore(path)
    record = await reopened._read_root_turn_for_executor(
      job_id="job-original", turn_id="turn-original", tenant_storage_id="alice",
    )
    assert record is not None
    assert await reopened.read_policy_generation(
      generation_ref=record.policy_generation_ref,
    ) == original
    current = await reopened.current_policy_generation()
    assert current == changed
    assert current is not None
    assert verify_policy_generation(
      record.policy_generation_ref, original.generation_ref,
    ) is None
    mismatch = verify_policy_generation(
      record.policy_generation_ref, current.generation_ref,
    )
    assert isinstance(mismatch, PolicyGenerationMismatch)
    assert mismatch.code == "approval_policy_generation_mismatch"
    assert mismatch.recorded == original.generation_ref
    assert mismatch.current == changed.generation_ref
    # Resolving/re-recording historical A must not move the current B pointer.
    assert await reopened.record_policy_generation(
      **identity, make_current=False,
    ) == original
    assert await reopened.current_policy_generation() == changed

  asyncio.run(case())


def test_policy_generation_records_and_current_pointer_survive_reopen(
  tmp_path: Path,
) -> None:
  async def case() -> None:
    path = tmp_path / "approvals.sqlite3"
    store = SQLiteApprovalStore(path)
    assert await store.current_policy_generation() is None
    assert await store.read_policy_generation(generation_ref="absent") is None
    generation_a = await store.record_policy_generation(
      policy_id="single-user", policy_version="4", policy_bundle_hash="a" * 64,
    )
    generation_b = await store.record_policy_generation(
      policy_id="single-user", policy_version="4", policy_bundle_hash="b" * 64,
    )
    reopened = SQLiteApprovalStore(path)
    assert await reopened.current_policy_generation() == generation_b
    assert await reopened.read_policy_generation(
      generation_ref=generation_a.generation_ref,
    ) == generation_a
    assert await reopened.record_policy_generation(
      policy_id="single-user", policy_version="4", policy_bundle_hash="b" * 64,
    ) == generation_b
    assert await reopened.current_policy_generation() == generation_b

  asyncio.run(case())


def test_existing_approval_store_gains_root_tables_without_rebinding_calls(
  tmp_path: Path,
) -> None:
  async def case() -> None:
    path = tmp_path / "approvals.sqlite3"
    store = SQLiteApprovalStore(path)
    request, prepared = await _persist_binding(store)
    # Retain a real pre-root schema/database: only the two new empty tables go.
    with store._connection() as conn:
      conn.execute("DROP TABLE approval_root_turns")
      conn.execute("DROP TABLE approval_policy_generations")
    reopened = SQLiteApprovalStore(path)
    binding, decision = await reopened._read_executable_binding_for_executor(
      job_id="job-original", tool_call_id=request.tool_call_id,
      nonce="nonce-original", tenant_storage_id="alice",
    )
    assert binding.prepared_input == prepared.prepared_input
    assert decision.approval_id == request.approval_id
    await reopened.record_root_turn(**await _root_turn_arguments(reopened))
    await reopened.link_root_turn_executable_binding(
      job_id="job-original", turn_id="turn-original",
      tool_call_id=request.tool_call_id, nonce="nonce-original",
      tenant_storage_id="alice",
    )
    cold = SQLiteApprovalStore(path)
    assert await cold._locate_root_turn_for_approval(
      approval_id=request.approval_id, tenant_storage_id="alice",
    ) == ("job-original", request.tool_call_id, "nonce-original", "turn-original")

  asyncio.run(case())
