from __future__ import annotations

import logging
import time
from typing import (
  TYPE_CHECKING,
  Any,
  Awaitable,
  Callable,
  Dict,
  List,
)

from .capability_execution import BoundCapabilityExecution, CapabilityExecutionResolver
from .multi_user.billing import (
  BillingMode,
  SessionUsageSummary,
  UsageEvent,
  UsageState,
  UsageTotals as _UsageTotals,
)
from .runner_auth import merge_refreshed_auth_config as _merge_refreshed_auth_config
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

if TYPE_CHECKING:
  from pathlib import Path

  from .capability_execution import ModelFallbackHop
  from .event_log import EventLog
  from .multi_user.billing import _UsageAggregator
  from .runner_tool_execution import AgentRunnerDispatcher

log = logging.getLogger("agent_gateway.runner")


class RunnerHooksLifecycleMixin:
  if TYPE_CHECKING:
    _aggregator: _UsageAggregator
    _auth_config: Dict[str, Any]
    _billing_mode: BillingMode
    _dispatcher: AgentRunnerDispatcher
    _channel: str | None
    _context_surface_records: Callable[[], List[Dict[str, Any]]]
    _full_session_id: str
    _log: EventLog
    _on_before_stream_complete: (
      Callable[..., Awaitable[None] | None] | None
    )
    _on_late_usage_event: (
      Callable[[UsageEvent], Awaitable[None] | None] | None
    )
    _on_metric: Callable[[str, int], None] | None
    _on_session_summary: (
      Callable[[SessionUsageSummary], Awaitable[None] | None] | None
    )
    _on_tool_result: (
      Callable[
        [ToolResultContext],
        Awaitable[List[Dict[str, Any]] | None] | List[Dict[str, Any]] | None,
      ]
      | None
    )
    _on_tool_timing: Callable[..., None] | None
    _on_tool_timing_accepts_context_surfaces: bool
    _on_tool_timing_accepts_user_id: bool
    _on_usage: Callable[[UsageEvent], Awaitable[None] | None] | None
    _parent_turn_id: str | None
    _rate_table_version: str
    _request_id: str
    _sid: str
    _summary_emitted: bool
    _model_fallback_hop: ModelFallbackHop | None
    _usage_ledger_dlq_path: Path
    _usage_user_id: str

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

  def _apply_refreshed_auth_config(self, config: Dict[str, Any], refreshed: Dict[str, Any]) -> bool:
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
    refreshed_execution = CapabilityExecutionResolver.rebind_rotated_credential(
      execution,
      refreshed,
      merged_auth_config=merged,
    )
    if refreshed_execution is None:
      return False
    self._install_capability_execution(config, refreshed_execution)
    return True

  def _install_capability_execution(
    self,
    config: Dict[str, Any],
    execution: BoundCapabilityExecution,
  ) -> None:
    """Make a rebound execution the runner's and its dispatcher's.

    Credential rotation and a model-fallback hop both land here: the request
    config, provider adapter, secret boundary and tool-capture bind all follow
    the one execution for the remaining attempts.
    """
    run_config = normalized_run_config(
      execution.auth_config,
      upstream_model=execution.bind.upstream_model,
      effort=execution.bind.effort,
    )
    self._dispatcher.replace_capability_bind(execution.bind)
    config.clear()
    config.update(run_config)
    self._auth_config.clear()
    self._auth_config.update(execution.auth_config)
    self._provider = execution.provider
    self._capability_execution = execution
    current_boundary = getattr(self, "_secret_boundary", None)
    self._secret_boundary = (
      current_boundary.extended_for_capability_execution(
        execution
      )
      if isinstance(current_boundary, SecretBoundary)
      else SecretBoundary.from_capability_execution(execution)
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

  def _build_usage_event(self, *, model: str, usage_totals: _UsageTotals) -> UsageEvent:
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
      selection_provenance=self._usage_selection_provenance(),
    )

  def _usage_selection_provenance(self) -> dict[str, Any] | None:
    """V3 provenance for this call, or None when the runner captured none."""
    context = self._dispatcher.capture_spawning_turn().context
    if context is None:
      return None
    provenance = context.selection_provenance
    if provenance is None:
      return None
    recorded = provenance.model_dump(mode="json")
    hop = self._model_fallback_hop
    reason = getattr(self, "_model_fallback_reason", None)
    if hop is not None and reason in ("usage_limit", "provider_outage", "credential_denied"):
      recorded["fallback"] = {
        "primary_model_key": hop.primary_model_key,
        "role": hop.role,
        "chain_index": hop.chain_index,
        "reason": reason,
      }
    else:
      recorded["fallback"] = None
    return recorded

  async def _call_on_usage(
    self, usage_event: UsageEvent, *, usage_state: UsageState = "succeeded"
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

  async def _call_on_session_summary(
    self,
    summary: SessionUsageSummary,
    *,
    reconciled: SessionUsageSummary | None = None,
  ) -> None:
    # Only the root run reconciles commercial usage, against the whole tree.
    await _runner_attr(self, "_call_session_summary_hook", _call_session_summary_hook)(
      self._on_session_summary,
      summary,
      log_session_id=self._sid,
      logger=log,
      commercial_usage_producer=(
        None
        if getattr(self, "_usage_parent", None) is not None
        else getattr(self, "_commercial_usage_producer", None)
      ),
      reconciled=reconciled,
      emit_metric=self._call_metric,
    )

  def _estimate_usage_cost(self, model: str, usage_totals: _UsageTotals):
    return _runner_attr(self, "_estimate_usage_cost", _estimate_usage_cost)(self._provider, model, usage_totals)
