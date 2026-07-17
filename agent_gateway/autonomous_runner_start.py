from __future__ import annotations

import asyncio
import functools
import json
import os
import time
from typing import Any

from .autonomous_runner_claims import get_agent_api_claim_ttl_seconds, sign_user_claim
from .autonomous_runner_commands import normalize_autonomous_profile, normalize_max_budget_usd
from .artifact_paths import canonicalize_ticker
from .autonomous_runner_state import (
  AutonomousTask,
  _fallback_identity_payload,
  _normalize_dispatch_scope,
  _normalize_identity_aliases,
  _positive_int,
  _runtime_attr,
  _user_identity_api,
)

_SPAWN_CLEANUP_GRACE_SEC = 1.0


def _with_run_mutation_lock(func):
  @functools.wraps(func)
  async def guarded(self, *args, **kwargs):
    async with self.run_mutation_lock:
      return await func(self, *args, **kwargs)

  return guarded


def _asyncio_module() -> Any:
  return _runtime_attr("asyncio", asyncio)


def _start_identity_payload(
  *,
  raw_user_id: str,
  user_email: str | None,
  owner_user_id: str | None,
  user_slug: str | None,
  risk_user_id: int | None,
  user_aliases: list[str] | tuple[str, ...] | None,
  identity_status: str | None,
) -> dict[str, Any]:
  explicit_owner = str(owner_user_id or "").strip() or None
  normalized_slug = str(user_slug or "").strip() or None
  normalized_risk = _positive_int(risk_user_id)

  if explicit_owner is None:
    api = _user_identity_api()
    resolver = getattr(api, "resolve_canonical_user_identity", None) if api is not None else None
    if callable(resolver):
      try:
        identity = resolver(
          raw_user_id,
          risk_user_id=risk_user_id,
          user_email=user_email,
          mapped_slug=normalized_slug,
          allow_legacy_fallback=True,
        )
        return {
          "owner_user_id": str(identity.owner_user_id),
          "raw_user_id": raw_user_id,
          "user_slug": identity.user_slug,
          "risk_user_id": int(identity.risk_user_id),
          "user_aliases": _normalize_identity_aliases(
            identity.owner_user_id,
            identity.raw_user_id,
            identity.user_slug,
            identity.user_email,
            identity.aliases,
            user_aliases,
          ),
          "identity_status": identity_status or str(identity.identity_status),
        }
      except ValueError as exc:
        raise ValueError(f"Unable to resolve canonical autonomous user identity for {raw_user_id!r}") from exc

  fallback_owner = explicit_owner or (str(normalized_risk) if normalized_risk is not None else raw_user_id)
  fallback_slug = normalized_slug
  if fallback_slug is None and raw_user_id != fallback_owner and not raw_user_id.isdecimal():
    fallback_slug = raw_user_id
  return _fallback_identity_payload(
    user_id=fallback_owner,
    user_email=user_email,
    identity_status=identity_status
    or ("risk_user_id_authoritative" if normalized_risk is not None else "legacy_user_id_fallback"),
    risk_user_id=normalized_risk,
    owner_user_id=fallback_owner,
    raw_user_id=raw_user_id,
    user_slug=fallback_slug,
    user_aliases=list(user_aliases or ()),
  )


class AutonomousRegistryStartMixin:
  async def _reserve_slot(self) -> None:
    async with self._slot_lock:
      if self._reserved_slots >= self._max_running:
        raise RuntimeError(f"Autonomous concurrency limit reached ({self._max_running})")
      self._reserved_slots += 1

  async def _release_slot(self, record: AutonomousTask | None = None) -> None:
    async with self._slot_lock:
      if record is None:
        self._reserved_slots = max(0, self._reserved_slots - 1)
        return
      if not record.slot_reserved:
        return
      record.slot_reserved = False
      self._reserved_slots = max(0, self._reserved_slots - 1)

  async def _await_cleanup(self, cleanup_coro) -> None:
    asyncio_module = _asyncio_module()
    cleanup_task = asyncio_module.create_task(cleanup_coro)
    try:
      await asyncio_module.shield(cleanup_task)
    except asyncio_module.CancelledError:
      await cleanup_task
      raise

  async def _terminate_unowned_process(self, record: AutonomousTask | None) -> None:
    proc = None if record is None else record.proc
    if proc is None or proc.returncode is not None:
      return
    try:
      proc.terminate()
    except ProcessLookupError:
      return
    asyncio_module = _asyncio_module()
    try:
      await asyncio_module.wait_for(
        proc.wait(),
        timeout=_runtime_attr("_SPAWN_CLEANUP_GRACE_SEC", _SPAWN_CLEANUP_GRACE_SEC),
      )
    except asyncio_module.TimeoutError:
      try:
        proc.kill()
      except ProcessLookupError:
        pass
      await proc.wait()

  async def _cleanup_uncommitted_start(
    self,
    *,
    task_id: str,
    record: AutonomousTask | None,
    log_handle: Any | None,
  ) -> None:
    self._tasks.pop(task_id, None)
    await self._terminate_unowned_process(record)
    if log_handle is not None:
      log_handle.close()
    if record is not None:
      record.log_handle = None
      if record.events_tail_task is not None and not record.events_tail_task.done():
        record.events_tail_task.cancel()
        await _asyncio_module().gather(record.events_tail_task, return_exceptions=True)
      await self._release_slot(record)
    else:
      await self._release_slot()
    if record is not None and record.tool_result_spill_dir is not None:
      self._remove_registered_tool_result_spill_dir(
        task_id,
        record.tool_result_spill_dir,
      )
    self._delete_task_manifest(task_id)

  @_with_run_mutation_lock
  async def start(
    self,
    *,
    profile: str,
    mode: str,
    user_id: str,
    user_email: str | None,
    owner_user_id: str | None = None,
    user_slug: str | None = None,
    risk_user_id: int | None = None,
    user_aliases: list[str] | tuple[str, ...] | None = None,
    identity_status: str | None = None,
    control_run_id: str | None = None,
    task: str | None = None,
    skill: str | None = None,
    context: str | None = None,
    ticker: str | None = None,
    max_budget_usd: float | None = None,
    channel: str | None = None,
    dev_mode: bool = False,
    dispatch_scope: dict[str, Any] | None = None,
    resumed_from: str | None = None,
    schedule_id: str | None = None,
    schedule_name: str | None = None,
  ) -> dict[str, Any]:
    await self._reserve_slot()
    task_id = self._next_task_id()
    control_run_id = control_run_id or task_id
    log_handle = None
    record: AutonomousTask | None = None
    ownership_transferred = False
    try:
      raw_user_id = str(user_id or "").strip()
      if not raw_user_id:
        raise ValueError("user_id is required")
      identity = _start_identity_payload(
        raw_user_id=raw_user_id,
        user_email=user_email,
        owner_user_id=owner_user_id,
        user_slug=user_slug,
        risk_user_id=risk_user_id,
        user_aliases=user_aliases,
        identity_status=identity_status,
      )
      normalized_owner_user_id = str(identity["owner_user_id"])
      normalized_user_slug = identity["user_slug"]
      normalized_risk_user_id = int(identity["risk_user_id"])
      normalized_identity_status = str(identity["identity_status"])
      aliases = list(identity["user_aliases"])
      normalized_dispatch_scope = _normalize_dispatch_scope(dispatch_scope)
      if dispatch_scope is not None and normalized_dispatch_scope is None:
        raise ValueError("dispatch_scope must be a redacted portfolio dispatch scope")
      normalized_max_budget_usd = normalize_max_budget_usd(max_budget_usd)
      cmd = self._build_cmd(
        profile=profile,
        mode=mode,
        task=task,
        skill=skill,
        context=context,
        ticker=ticker,
        dev_mode=dev_mode,
        max_budget_usd=normalized_max_budget_usd,
      )
      normalized_mode = mode.strip().lower()
      effective_dev_mode = bool(dev_mode or normalized_mode == "task")
      log_path = self._log_dir / f"{task_id}.log"
      events_path = self._log_dir / f"{task_id}.events.jsonl"
      operator_inbox_path = self._log_dir / f"{task_id}.operator-messages.jsonl"
      approval_decisions_path = self._log_dir / f"{task_id}.approval-decisions.jsonl"
      self._log_dir.mkdir(parents=True, exist_ok=True)
      events_path.write_text("", encoding="utf-8")
      operator_inbox_path.write_text("", encoding="utf-8")
      approval_decisions_path.write_text("", encoding="utf-8")
      log_handle = log_path.open("wb")
      normalize_profile = _runtime_attr("normalize_autonomous_profile", normalize_autonomous_profile)
      time_module = _runtime_attr("time", time)
      record = AutonomousTask(
        task_id=task_id,
        control_run_id=control_run_id,
        user_id=normalized_owner_user_id,
        user_email=user_email,
        profile=normalize_profile(profile),
        mode=mode.strip().lower(),
        task=task.strip() if isinstance(task, str) and task.strip() else None,
        skill=skill.strip() if isinstance(skill, str) and skill.strip() else None,
        context=context.strip() if isinstance(context, str) and context.strip() else None,
        ticker=canonicalize_ticker(ticker) if isinstance(ticker, str) and ticker.strip() else None,
        channel=channel.strip().lower() if isinstance(channel, str) and channel.strip() else None,
        dev_mode=effective_dev_mode,
        dispatch_scope=normalized_dispatch_scope,
        cmd=cmd,
        log_path=log_path,
        events_path=events_path,
        operator_inbox_path=operator_inbox_path,
        approval_decisions_path=approval_decisions_path,
        started_at=time_module.time(),
        max_budget_usd=normalized_max_budget_usd,
        state="starting",
        log_handle=log_handle,
        slot_reserved=True,
        event_lines=[],
        owner_user_id=normalized_owner_user_id,
        raw_user_id=raw_user_id,
        user_slug=normalized_user_slug,
        risk_user_id=normalized_risk_user_id,
        user_aliases=aliases,
        identity_status=normalized_identity_status,
        resumed_from=resumed_from.strip() if isinstance(resumed_from, str) and resumed_from.strip() else None,
        schedule_id=schedule_id.strip() if isinstance(schedule_id, str) and schedule_id.strip() else None,
        schedule_name=schedule_name.strip() if isinstance(schedule_name, str) and schedule_name.strip() else None,
        tool_result_spill_dir=self._expected_tool_result_spill_dir(task_id),
      )
      self._attach_manifest_tracking(record)
      self._tasks[task_id] = record
      asyncio_module = _asyncio_module()
      record.events_tail_task = asyncio_module.create_task(self._tail_events_file(task_id))

      if not self._write_task_manifest(record, checked=True):
        record.tool_result_spill_dir = None
      else:
        try:
          record.tool_result_spill_dir.mkdir(mode=0o700)
        except Exception as exc:
          raise RuntimeError(f"autonomous spill directory setup failed: {exc}") from exc

      os_module = _runtime_attr("os", os)
      env = dict(os_module.environ)
      env["PYTHONUNBUFFERED"] = "1"
      hmac_key = os_module.getenv("AGENT_API_USER_CLAIM_HMAC_KEY", "").strip()
      if not hmac_key:
        raise RuntimeError(
          "AGENT_API_USER_CLAIM_HMAC_KEY required for autonomous dispatch. "
          "Set it in the gateway env (.env or process env)."
        )
      sign_claim = _runtime_attr("sign_user_claim", sign_user_claim)
      get_claim_ttl = _runtime_attr(
        "get_agent_api_claim_ttl_seconds",
        get_agent_api_claim_ttl_seconds,
      )
      claim_env = sign_claim(
        hmac_key,
        user_id=record.owner_user_id or record.user_id,
        user_email=user_email,
        ttl_seconds=get_claim_ttl(),
      )
      env.update(claim_env)
      env["AUTONOMOUS_USER_ID"] = record.owner_user_id or record.user_id
      env["AUTONOMOUS_RAW_USER_ID"] = record.raw_user_id or raw_user_id
      env["AUTONOMOUS_USER_SLUG"] = record.user_slug or ""
      env["AUTONOMOUS_USER_EMAIL"] = user_email or ""
      env["AGENT_AUTONOMOUS_EVENTS_PATH"] = str(events_path)
      env["AGENT_AUTONOMOUS_OPERATOR_INBOX_PATH"] = str(operator_inbox_path)
      env["AGENT_AUTONOMOUS_APPROVAL_DECISIONS_PATH"] = str(approval_decisions_path)
      if record.tool_result_spill_dir is not None:
        env["AGENT_AUTONOMOUS_TOOL_RESULT_SPILL_DIR"] = str(record.tool_result_spill_dir)
      env["AGENT_AUTONOMOUS_GATEWAY_SESSION_ID"] = (
        f"agent-control:{control_run_id}:{int(record.started_at)}"
      )
      env["AGENT_AUTONOMOUS_CONTROL_RUN_ID"] = control_run_id
      env["AGENT_AUTONOMOUS_CONTROL_CHANNEL"] = record.channel or ""
      if record.dispatch_scope is not None:
        env["AGENT_AUTONOMOUS_DISPATCH_SCOPE_JSON"] = json.dumps(
          record.dispatch_scope,
          sort_keys=True,
          separators=(",", ":"),
        )
      if record.dev_mode:
        env[f"{record.profile.upper().replace('-', '_')}_DEV_MODE"] = "true"
      if self._approval_db_path is not None:
        env["AGENT_AUTONOMOUS_APPROVALS_DB_PATH"] = str(self._approval_db_path)
      record.proc = await asyncio_module.create_subprocess_exec(
        *cmd,
        cwd=str(self._api_dir),
        stdin=asyncio_module.subprocess.DEVNULL,
        stdout=log_handle,
        stderr=asyncio_module.subprocess.STDOUT,
        env=env,
      )

      assert record is not None
      record.state = "running"
      if not self._write_task_manifest(record, checked=True):
        raise RuntimeError("failed to commit running autonomous task manifest")
      record.reaper_task = asyncio_module.create_task(self._reap(task_id))
      await self._publish_run_state(record, "running")
      ownership_transferred = True
      return self._start_payload(record)
    except OSError as exc:
      raise RuntimeError(f"spawn failed: {exc}") from exc
    finally:
      if not ownership_transferred:
        await self._await_cleanup(
          self._cleanup_uncommitted_start(
            task_id=task_id,
            record=record,
            log_handle=log_handle,
          )
        )


__all__ = [
  "AutonomousRegistryStartMixin",
  "_SPAWN_CLEANUP_GRACE_SEC",
]
