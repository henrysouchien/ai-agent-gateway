from __future__ import annotations

import logging
import time
from typing import Any, Dict, List

from .auth import ProviderCredentialFailure
from .capability_execution import BoundCapabilityExecution
from .multi_user.billing import SessionUsageSummary, UsageEvent
from .runner_auth import (
  call_credential_refresher as _call_credential_refresher,
  merge_refreshed_auth_config as _merge_refreshed_auth_config,
)
from .secret_boundary import SecretBoundary
from .runner_callbacks import (
  call_before_stream_complete_hook as _call_before_stream_complete_hook,
  call_metric_hook as _call_metric_hook,
  call_tool_timing_hook as _call_tool_timing_hook,
  call_tool_result_hook as _call_tool_result_hook,
)
from .runner_background_lifecycle import _runner_module_attr
from .runner_session_lifecycle import _runner_attr
from .runner_state import ToolResultContext, normalized_run_config
from .runner_usage import (
  build_usage_event as _build_usage_event,
  call_late_usage_event_hook as _call_late_usage_event_hook,
  call_session_summary_hook as _call_session_summary_hook,
  call_usage_event_hook as _call_usage_event_hook,
  estimate_usage_cost as _estimate_usage_cost,
  usage_delta as _usage_delta,
  usage_has_tokens as _usage_has_tokens,
)


log = logging.getLogger("agent_gateway.runner")


class RunnerHooksLifecycleMixin:
  async def _call_on_tool_result(self, ctx: ToolResultContext) -> List[Dict[str, Any]]:
    return await _runner_attr(self, "_call_tool_result_hook", _call_tool_result_hook)(
      self._on_tool_result,
      ctx,
      log_session_id=getattr(self, "_sid", getattr(self, "_full_session_id", "")),
      logger=log,
    )

  async def _call_on_before_stream_complete(self, terminal_event: Dict[str, Any] | None = None) -> None:
    if self._on_before_stream_complete is None:
      return
    await _runner_attr(self, "_call_before_stream_complete_hook", _call_before_stream_complete_hook)(
      self._on_before_stream_complete,
      self._log,
      terminal_event,
      log_session_id=getattr(self, "_sid", getattr(self, "_full_session_id", "")),
      logger=log,
    )


  def _call_on_tool_timing(
    self,
    *,
    tool_name: str,
    server: str | None,
    duration_ms: int,
    is_error: bool,
    result_bytes: int,
    tool_call_id: str | None = None,
    request_id: str | None = None,
  ) -> None:
    _runner_attr(self, "_call_tool_timing_hook", _call_tool_timing_hook)(
      self._on_tool_timing,
      accepts_user_id=self._on_tool_timing_accepts_user_id,
      accepts_context_surfaces=self._on_tool_timing_accepts_context_surfaces,
      accepts_tool_call_id=getattr(self, "_on_tool_timing_accepts_tool_call_id", False),
      accepts_request_id=getattr(self, "_on_tool_timing_accepts_request_id", False),
      session_id=self._full_session_id,
      log_session_id=self._sid,
      user_id=self._usage_user_id,
      context_surfaces=self._context_surface_records(),
      tool_call_id=tool_call_id,
      request_id=request_id,
      tool_name=tool_name,
      server=server,
      duration_ms=duration_ms,
      is_error=is_error,
      result_bytes=result_bytes,
      logger=log,
    )

  def _call_metric(self, name: str, value: int = 1) -> None:
    _runner_attr(self, "_call_metric_hook", _call_metric_hook)(
      self._on_metric,
      name=name,
      value=value,
      log_session_id=self._sid,
      logger=log,
    )

  async def _call_credential_refresher(self, failure: ProviderCredentialFailure) -> Dict[str, Any] | None:
    return await _runner_attr(self, "_call_credential_refresher", _call_credential_refresher)(
      self._on_credential_failure,
      failure,
      emit_metric=self._call_metric,
      log_session_id=self._sid,
      logger=log,
    )

  def _apply_refreshed_auth_config(self, config: Dict[str, Any], refreshed: Dict[str, Any]) -> None:
    execution = getattr(self, "_capability_execution", None)
    if not isinstance(execution, BoundCapabilityExecution):
      raise RuntimeError(
        "credential refresh requires an immutable capability execution"
      )
    execution.validate()
    active_config = dict(self._auth_config)
    if getattr(self, "_billing_mode", None):
      active_config["billing_mode"] = self._billing_mode
    if getattr(self, "_rate_table_version", None):
      active_config["rate_table_version"] = self._rate_table_version
    merged = _runner_attr(self, "_merge_refreshed_auth_config", _merge_refreshed_auth_config)(active_config, refreshed)
    refreshed_execution = BoundCapabilityExecution(
      bind=execution.bind,
      registry=execution.registry,
      adapter=execution.adapter,
      auth_config=merged,
    )
    refreshed_run_config = normalized_run_config(
      refreshed_execution.auth_config,
      upstream_model=refreshed_execution.bind.upstream_model,
      effort=refreshed_execution.bind.effort,
    )
    config.clear()
    config.update(refreshed_run_config)
    self._auth_config.clear()
    self._auth_config.update(refreshed_execution.auth_config)
    self._provider = refreshed_execution.provider
    self._capability_execution = refreshed_execution
    current_boundary = getattr(self, "_secret_boundary", None)
    self._secret_boundary = (
      current_boundary.extended_for_capability_execution(
        refreshed_execution
      )
      if isinstance(current_boundary, SecretBoundary)
      else SecretBoundary.from_capability_execution(refreshed_execution)
    )
    bind_dispatcher_boundary = getattr(
      getattr(self, "_dispatcher", None),
      "bind_secret_boundary",
      None,
    )
    if callable(bind_dispatcher_boundary):
      bind_dispatcher_boundary(self._secret_boundary)

  @staticmethod
  def _usage_has_tokens(usage_totals: Dict[str, int]) -> bool:
    return _runner_module_attr("_usage_has_tokens", _usage_has_tokens)(usage_totals)

  @staticmethod
  def _usage_delta(before: Dict[str, int], after: Dict[str, int]) -> Dict[str, int]:
    return _runner_module_attr("_usage_delta", _usage_delta)(before, after)

  def _build_usage_event(self, *, model: str, usage_totals: Dict[str, int]) -> UsageEvent:
    cost = self._estimate_usage_cost(model, usage_totals)
    return _runner_attr(self, "_build_usage_event", _build_usage_event)(
      user_id=self._usage_user_id,
      session_id=self._full_session_id,
      request_id=self._request_id,
      parent_turn_id=self._parent_turn_id,
      timestamp=_runner_attr(self, "time", time).time(),
      model=model,
      provider_name=getattr(self._provider, "name", None),
      usage_totals=usage_totals,
      cost_total=float(cost.total),
      rate_table_version=self._rate_table_version,
      billing_mode=self._billing_mode,
      channel=self._channel,
    )

  async def _call_on_usage(
    self, usage_event: UsageEvent, *, usage_state: str = "succeeded"
  ) -> None:
    await _runner_attr(self, "_call_usage_event_hook", _call_usage_event_hook)(
      self._aggregator,
      usage_event,
      is_summary_emitted=lambda: self._summary_emitted,
      on_usage=self._on_usage,
      on_late_usage_event=self._on_late_usage_event,
      emit_metric=self._call_metric,
      dlq_path=self._usage_ledger_dlq_path,
      log_session_id=self._sid,
      logger=log,
      commercial_usage_producer=getattr(self, "_commercial_usage_producer", None),
      usage_state=usage_state,
    )

  async def _call_on_late_usage_event(self, usage_event: UsageEvent) -> None:
    await _runner_attr(self, "_call_late_usage_event_hook", _call_late_usage_event_hook)(
      self._on_late_usage_event,
      usage_event,
      log_session_id=self._sid,
      logger=log,
    )

  async def _call_on_session_summary(self, summary: SessionUsageSummary) -> None:
    await _runner_attr(self, "_call_session_summary_hook", _call_session_summary_hook)(
      self._on_session_summary,
      summary,
      log_session_id=self._sid,
      logger=log,
      commercial_usage_producer=getattr(self, "_commercial_usage_producer", None),
      emit_metric=self._call_metric,
    )

  def _estimate_usage_cost(self, model: str, usage_totals: Dict[str, int]):
    return _runner_attr(self, "_estimate_usage_cost", _estimate_usage_cost)(self._provider, model, usage_totals)
