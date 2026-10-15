from __future__ import annotations

import asyncio
import copy
import logging
import sys
import time
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional, Tuple, Union

from .auth import ProviderCredentialFailure
from .providers.anthropic import referenced_tool_names
from .providers.anthropic_oauth import AnthropicSourceError
from .capability_execution import CapabilityExecutionResolver, FallbackSkip, ModelFallbackHop
from model_authority.binding import validate_reported_identity
from .providers import ModelInfo
from .rate_limit import local_queue_left_at, locally_queued
from .providers.anthropic_helpers import _format_anthropic_rejection_detail
from model_authority.thinking import EffortResolution, ThinkingLevel, parse_effort
from .runner_introspection import format_exc as _format_exc
from .runner_limits import (
  effective_compaction_trigger as _effective_compaction_trigger,
  model_context_window,
  system_prompt_estimate_text,
  token_breakdown_snapshot as _token_breakdown_snapshot,
  token_estimate_snapshot,
)
from .runner_prompt_rules import (
  messages_require_tool_only_turns as _messages_require_tool_only_turns,
  system_prompt_requires_tool_only_turns as _system_prompt_requires_tool_only_turns,
)
from .runner_session_events import error_with_provider_projection
from .runner_cleanup import attach_cleanup_failure
from .runner_session_lifecycle import _runner_attr
from .commercial_work_start import commercial_budget_stop_fields
from .runner_state import StreamTurnFailure, StreamTurnResult, select_run_max_tokens
from .runner_streaming import (
  STREAM_STALL_TIMEOUT,
  STREAM_THINKING_STALL_TIMEOUT,
  classify_guard_outcome,
  effective_stream_stall_timeout,
  observed_thinking_in_messages,
  thinking_level,
)
from .runner_usage import (
  apply_message_start_usage as _apply_message_start_usage,
  apply_usage_update as _apply_usage_update,
  usage_snapshot,
  usage_delta_state as _usage_delta_state,
)

if TYPE_CHECKING:
  from .capability_execution import BoundCapabilityExecution
  from .containment_settings import ContainmentSettings
  from .multi_user.billing import UsageEvent, UsageState
  from .providers import ModelProvider
  from .runner_hooks_lifecycle import _UsageTotals
  from .runner_tool_execution import AgentRunnerDispatcher
  from .session import RequestToolPrefix



log = logging.getLogger("agent_gateway.runner")
STREAM_GUARD_POLL_INTERVAL = 2.0
STREAM_PROGRESS_LOG_INTERVAL = 60.0
STREAM_RETRY_MAX = 3
STREAM_RETRY_DELAY = 2.0
STREAM_RETRY_BACKOFF = 2.0


def _runner_module_attr(name: str, fallback: Any) -> Any:
  module = sys.modules.get("agent_gateway.runner")
  if module is None:
    return fallback
  return getattr(module, name, fallback)


def _format_error_with_rejection_detail(
  formatted_error: str,
  exc: Exception,
) -> str:
  rejection_detail = _format_anthropic_rejection_detail(exc)
  if rejection_detail is None:
    return formatted_error
  return f"{formatted_error} | {rejection_detail}"


class RunnerStreamTurnMixin:
  _first_text_at: float | None = None
  # This runner's last fallback hop: the chain position the next hop walks
  # on from. The next turn's runner starts on the primary with none.
  _model_fallback_hop: ModelFallbackHop | None = None
  _model_fallback_reason: Literal["usage_limit", "credential_denied", "provider_outage"] | None = None

  if TYPE_CHECKING:
    _billing_mode: Literal["byok", "metered"]
    _capability_execution: BoundCapabilityExecution
    _compaction_instructions: str | None
    _compaction_trigger: int | None
    _containment: ContainmentSettings
    _disconnected: bool
    _dispatcher: AgentRunnerDispatcher
    _effort_resolution: EffortResolution | None
    _last_request_max_tokens: int
    _last_request_message_marker_position: (
      tuple[int, int] | None
    )
    _last_request_system_blocks: tuple[tuple[str, bool], ...]
    _last_request_wire_tools: list[dict[str, Any]]
    _last_request_advertised_tool_names: frozenset[str]
    _tool_prefix: RequestToolPrefix
    _max_tokens_override: int | None
    _provider: ModelProvider
    _sid: str
    _stream_stall_timeout: float | None

    def _append(self, event: Dict[str, Any]) -> Any | None: ...

    def _apply_refreshed_auth_config(
      self,
      config: Dict[str, Any],
      refreshed: Dict[str, Any],
    ) -> bool: ...

    def _build_usage_event(
      self,
      *,
      model: str,
      usage_totals: _UsageTotals,
    ) -> UsageEvent: ...

    def _call_metric(self, name: str, value: int = 1) -> None: ...

    def _install_capability_execution(
      self,
      config: Dict[str, Any],
      execution: BoundCapabilityExecution,
    ) -> None: ...

    async def _call_on_usage(
      self,
      usage_event: UsageEvent,
      *,
      usage_state: UsageState = "succeeded",
    ) -> None: ...

    async def _close_client(
      self,
      client: Any,
      timeout: float = 2.0,
    ) -> None: ...

    async def _emit_error_event(self, error: str) -> None: ...

    async def _emit_stream_retry_event(
      self,
      *,
      attempt: int,
      error: str,
      attempt_elapsed_s: float,
      backoff_s: float | None,
    ) -> None: ...

    def _set_client(self, client: Any) -> None: ...

    async def force_close(self, timeout: float = 2.0) -> None: ...

  @staticmethod
  def _thinking_level(enabled: bool) -> ThinkingLevel:
    return _runner_module_attr("thinking_level", thinking_level)(enabled)

  def _effective_stream_stall_timeout(
    self,
    *,
    config: Dict[str, Any],
    model_info: ModelInfo,
    max_tokens: int,
    effort_resolution: EffortResolution | None = None,
    current_messages: List[Dict[str, Any]] | None = None,
  ) -> float:
    observed_thinking = _runner_module_attr(
      "observed_thinking_in_messages",
      observed_thinking_in_messages,
    )(
      current_messages or [],
      model_info=model_info,
    )
    return _runner_attr(self, "effective_stream_stall_timeout", effective_stream_stall_timeout)(
      self._stream_stall_timeout,
      config=config,
      model_info=model_info,
      max_tokens=max_tokens,
      effort_resolution=effort_resolution,
      observed_thinking=observed_thinking,
      stream_stall_timeout_default=_runner_attr(self, "STREAM_STALL_TIMEOUT", STREAM_STALL_TIMEOUT),
      stream_thinking_stall_timeout_default=_runner_attr(
        self,
        "STREAM_THINKING_STALL_TIMEOUT",
        STREAM_THINKING_STALL_TIMEOUT,
      ),
    )

  @staticmethod
  def _classify_guard_outcome(
    guard_reason: tuple[str, str] | None,
    attempt: int,
    max_attempts: int,
  ) -> tuple[str, str, str]:
    return _runner_module_attr("classify_guard_outcome", classify_guard_outcome)(guard_reason, attempt, max_attempts)

  async def _finish_failed_stream_turn(self, client: Any, error: str) -> None:
    cancellation: asyncio.CancelledError | None = None
    try:
      await self._emit_error_event(error)
    except asyncio.CancelledError as exc:
      cancellation = exc
      raise
    finally:
      # Standalone settlement drains before re-raising cancellation; cleanup
      # must not replace that cancellation with a provider close failure.
      try:
        await self._close_client(client, timeout=5.0)
      except Exception as exc:
        if cancellation is None:
          raise
        _runner_attr(self, "log", log).warning(
          "[%s] client close after cancelled stream failed: %s",
          self._sid,
          attach_cleanup_failure(cancellation, exc),
        )

  async def _stream_turn(
    self,
    *,
    client: Any,
    config: Dict[str, Any],
    model_info: ModelInfo,
    system_prompt: Optional[Union[str, List[Tuple[str, bool]]]],
    current_messages: List[Dict[str, Any]],
    base_kwargs: Dict[str, Any],
    max_tokens: int,
    turn_count: int,
    turn_t0: float,
    turn_t0_mono: float,
    system_chars: int,
    tools_chars: int,
    usage_totals: _UsageTotals,
  ) -> Tuple[Any, StreamTurnResult] | StreamTurnFailure | None:
    asyncio_module = _runner_attr(self, "asyncio", asyncio)
    time_module = _runner_attr(self, "time", time)
    logger = _runner_attr(self, "log", log)
    cancelled_error_type = getattr(asyncio_module, "CancelledError", asyncio.CancelledError)
    first_completed = getattr(asyncio_module, "FIRST_COMPLETED", asyncio.FIRST_COMPLETED)
    stream_retry_max = _runner_attr(self, "STREAM_RETRY_MAX", STREAM_RETRY_MAX)
    stream_retry_delay = _runner_attr(self, "STREAM_RETRY_DELAY", STREAM_RETRY_DELAY)
    stream_retry_backoff = _runner_attr(self, "STREAM_RETRY_BACKOFF", STREAM_RETRY_BACKOFF)
    stream_guard_poll_interval = _runner_attr(self, "STREAM_GUARD_POLL_INTERVAL", STREAM_GUARD_POLL_INTERVAL)
    last_progress_at = time_module.monotonic()
    progress_events = 0
    guard_reason: tuple[str, str] | None = None

    def _bound_request_facts() -> tuple[ThinkingLevel, EffortResolution, float]:
      requested = parse_effort(
        config.get("effort"),
        field_name="capability_execution.effort",
      )
      if requested is None:
        raise ValueError(
          "native execution requires an explicitly bound effort"
        )
      bound_effort = self._capability_execution.bind.effort
      if requested.value != bound_effort:
        raise ValueError(
          "runtime effort does not match the immutable capability bind"
        )
      resolution = self._provider.resolve_effort(
        requested=requested,
        model=config["model"],
        model_info=model_info,
        max_tokens=max_tokens,
        auth_mode=config.get("auth_mode"),
        base_url=config.get("base_url") or config.get("baseURL"),
        compat=config.get("compat"),
      )
      if (
        resolution.requested != requested
        or resolution.effective != requested
      ):
        raise ValueError(
          (
            f"native execution cannot preserve bound effort "
            f"{requested.value!r} for "
            f"{self._capability_execution.bind.provider}:"
            f"{self._capability_execution.bind.upstream_model} "
            f"with max_tokens={max_tokens}"
          )
        )
      self._effort_resolution = resolution
      stall_timeout = self._effective_stream_stall_timeout(
        config=config,
        model_info=model_info,
        max_tokens=max_tokens,
        current_messages=current_messages,
        effort_resolution=resolution,
      )
      return requested, resolution, stall_timeout

    requested_effort, effort_resolution, _effective_stall_timeout = _bound_request_facts()

    def _history_fits(execution: BoundCapabilityExecution) -> bool:
      estimate = token_estimate_snapshot(
        system_text=system_prompt_estimate_text(system_prompt),
        messages=current_messages,
        tools=base_kwargs.get("tools") or [],
      )
      return estimate.est_total_tokens <= model_context_window(
        execution.provider.get_model_info(execution.bind.upstream_model),
        self._containment,
      )

    def _model_ref(execution: BoundCapabilityExecution) -> Dict[str, Any]:
      bind = execution.bind
      entry = execution.registry.models.get(bind.model_key)
      return {
        "model_key": bind.model_key,
        "label": entry.label if entry is not None else bind.model_key,
        "provider": bind.provider,
        "upstream_model": bind.upstream_model,
        "effort": bind.effort,
      }

    async def _hop_model_fallback(
      reason: Literal["usage_limit", "credential_denied", "provider_outage"],
      reset_at: float | None,
    ) -> bool:
      """Move the remaining attempts to the next chain entry (design §6.5).

      The hop happens at this attempt boundary: the failed attempt's partial
      output is already discarded and completed tool effects stay in the
      transcript, never replayed. Hop state is this runner's only; the next
      turn's runner binds the primary again.
      """

      nonlocal client, model_info, max_tokens, requested_effort, effort_resolution, _effective_stall_timeout
      # A reached reservation ends the job before the next provider request.
      # This is not a hop reason. rulings-index WA-ACCOUNT-BUDGET.
      stop = commercial_budget_stop_fields(
        getattr(self, "_commercial_usage_producer", None)
      )
      if stop is not None:
        setattr(self, "_request_boundary_stop_reason", "commercial_budget_reached")
        emit_interrupted = _runner_attr(self, "_emit_interrupted_event", None)
        if emit_interrupted is not None:
          await emit_interrupted(
            "commercial_budget_reached",
            extra_fields=stop,
          )
        return False
      context = self._dispatcher.capture_spawning_turn().context
      snapshot = context.selection_snapshot if context is not None else None
      if snapshot is None or context is None:
        return False
      primary = self._capability_execution
      skipped_reasons: list[FallbackSkip] = []
      hop = CapabilityExecutionResolver.rebind_model_fallback(
        primary, snapshot,
        selection_provenance=context.selection_provenance,
        history_fits=_history_fits,
        previous=self._model_fallback_hop,
        skipped_out=skipped_reasons,
      )
      if hop is None:
        reasons_summary = "; ".join(f"{s.model_key}: {s.code}" for s in skipped_reasons) if skipped_reasons else "none evaluated"
        logger.warning(
          "[%s] %s on %s with no eligible fallback chain entry (evaluated candidates: %s)",
          self._sid,
          reason,
          primary.bind.model_key,
          reasons_summary,
        )
        return False
      # The failed attempt's client belongs to the provider that created it;
      # the retry below builds the next one from the fallback's provider.
      await self._close_client(client, timeout=2.0)
      client = None
      self._install_capability_execution(config, hop.execution)
      self._model_fallback_hop = hop
      self._model_fallback_reason = reason
      model_info = self._provider.get_model_info(hop.execution.bind.upstream_model)
      max_tokens = select_run_max_tokens(
        config,
        max_tokens_override=self._max_tokens_override,
        model_info=model_info,
      ).value
      requested_effort, effort_resolution, _effective_stall_timeout = _bound_request_facts()
      tried_credential_refs.add(hop.execution.bind.credential_ref)
      logger.warning(
        "[%s] Model fallback on turn %d: %s -> %s (%s, role %s, chain index %d)",
        self._sid,
        turn_count,
        primary.bind.model_key,
        hop.execution.bind.model_key,
        reason,
        hop.role,
        hop.chain_index,
      )
      self._call_metric("gateway.model_fallback", 1)
      self._append({
        "type": "model_fallback",
        "from": _model_ref(primary),
        "to": _model_ref(hop.execution),
        "reason": reason,
        "reset_at": reset_at,
        "role": hop.role,
        "chain_index": hop.chain_index,
        "skipped": [
          {"model_key": skip.model_key, "code": skip.code}
          for skip in hop.skipped
        ],
      })
      return True

    def _make_params() -> tuple[Dict[str, Any], frozenset[str]]:
      normalized_messages = self._provider.normalize_messages(current_messages, model_info)
      request_tools = copy.deepcopy(base_kwargs.get("tools") or [])
      advertised_tool_names = frozenset(
        str(definition.get("name") or "").strip()
        for definition in request_tools
        if isinstance(definition, dict)
        and str(definition.get("name") or "").strip()
      )
      is_anthropic = getattr(self._provider, "name", None) == "anthropic"
      fork_kwargs: Dict[str, Any] = {}
      if is_anthropic:
        prefix = self._tool_prefix
        if not any(
          message.get("role") == "assistant" for message in normalized_messages
        ):
          # A conversation with no assistant turn yet is a new one (a pack
          # runs several skills, each from fresh history, on one session): it
          # anchors on its own first request, not on a predecessor's.
          prefix.anchor = None
          prefix.exposed.clear()
          prefix.framing_tool = None
        if prefix.anchor is None:
          prefix.framing_tool = None
          candidates: list[dict[str, Any]] = []
          mcp_client = getattr(self, "_mcp_client", None)
          if mcp_client is not None and hasattr(mcp_client, "get_tool_definitions"):
            try:
              raw_mcp_defs = mcp_client.get_tool_definitions()
              if isinstance(raw_mcp_defs, list):
                candidates.extend(raw_mcp_defs)
            except Exception:
              pass
          default_defs_fn = getattr(self, "_default_tool_definitions", None)
          if callable(default_defs_fn):
            try:
              raw_default_defs = default_defs_fn()
              if isinstance(raw_default_defs, list):
                candidates.extend(raw_default_defs)
            except Exception:
              pass
          for tool in candidates:
            if not isinstance(tool, dict):
              continue
            name = str(tool.get("name") or "").strip()
            if name and name not in advertised_tool_names:
              prefix.framing_tool = {
                key: value for key, value in tool.items() if key != "cache_control"
              }
              break
        fork_kwargs = {
          "fork_mode": bool(getattr(self, "_fork_mode", False)),
          "fork_marker_position": getattr(
            self,
            "_fork_marker_position",
            None,
          ),
          "tool_anchor": copy.deepcopy(prefix.anchor),
          "framing_tool": copy.deepcopy(prefix.framing_tool),
        }
        if prefix.anchor is not None:
          # A tool the visible history references keeps its definition even
          # after it leaves the advertised set, or the reference is a 400.  It
          # is on the wire for that reference only and confers no authority.
          referenced = referenced_tool_names(normalized_messages)
          request_tools.extend(
            copy.deepcopy(definition)
            for name, definition in prefix.exposed.items()
            if name in referenced and name not in advertised_tool_names
          )
      params = self._provider.build_request_params(
        model=config["model"],
        messages=normalized_messages,
        system_prompt=system_prompt,
        tools=request_tools,
        max_tokens=max_tokens,
        thinking_level=requested_effort,
        effort_resolution=effort_resolution,
        auth_mode=config["auth_mode"],
        base_url=config.get("base_url") or config.get("baseURL"),
        compat=config.get("compat"),
        compaction_trigger=_runner_attr(self, "_effective_compaction_trigger", _effective_compaction_trigger)(
          self._compaction_trigger,
          model_info,
          self._containment,
        ),
        compaction_instructions=self._compaction_instructions,
        **fork_kwargs,
      )
      if is_anthropic and self._tool_prefix.anchor is None:
        anchor = [
          tool for tool in params.get("tools") or []
          if not tool.get("defer_loading")
        ]
        self._tool_prefix.anchor = copy.deepcopy(anchor) if anchor else None
      rendered_system_blocks: list[tuple[str, bool]] = []
      for block in params.get("system") or []:
        if not isinstance(block, dict):
          continue
        text = str(block.get("text") or "")
        if text:
          rendered_system_blocks.append(
            (text, "cache_control" in block)
          )
      marker_locations: list[tuple[int, int]] = []
      for message_index, message in enumerate(
        params.get("messages") or []
      ):
        if not isinstance(message, dict):
          continue
        content = message.get("content")
        if not isinstance(content, list):
          continue
        marker_locations.extend(
          (message_index, block_index)
          for block_index, block in enumerate(content)
          if isinstance(block, dict)
          and "cache_control" in block
        )
      self._last_request_system_blocks = tuple(
        rendered_system_blocks
      )
      self._last_request_wire_tools = copy.deepcopy(
        params.get("tools") or []
      )
      self._last_request_advertised_tool_names = advertised_tool_names
      self._last_request_message_marker_position = (
        marker_locations[0] if len(marker_locations) == 1 else None
      )
      self._last_request_max_tokens = max_tokens
      return params, advertised_tool_names

    suppress_tool_turn_text = (
      _runner_attr(self, "_system_prompt_requires_tool_only_turns", _system_prompt_requires_tool_only_turns)(
        system_prompt
      )
      or _runner_attr(self, "_messages_require_tool_only_turns", _messages_require_tool_only_turns)(current_messages)
    )

    async def _consume_stream(params: Dict[str, Any], result: StreamTurnResult) -> None:
      nonlocal last_progress_at, progress_events
      first_turn = turn_count == 1
      logger.debug("[%s] Turn %d stream open", self._sid, turn_count)

      async for event in self._provider.stream(client, params):
        event_type = event.type
        if event_type != "heartbeat":
          last_progress_at = time_module.monotonic()
          progress_events += 1

        if event_type == "message_start":
          bind = self._capability_execution.bind
          usage_totals.update({
            "capability_bind": bind.to_json(),
          })
          if event.provider_reported_model is not None:
            usage_totals["provider_reported_model"] = validate_reported_identity(
              bind,
              event.provider_reported_model,
              registry=self._capability_execution.registry,
            )
          _runner_attr(self, "_apply_message_start_usage", _apply_message_start_usage)(
            usage_totals,
            input_tokens=event.input_tokens,
            cache_creation_tokens=event.cache_creation_tokens,
            cache_read_tokens=event.cache_read_tokens,
            provider_units=event.provider_units,
            provider_unit_deltas=event.provider_unit_deltas,
            provider_wait_ms=event.provider_wait_ms,
          )
          if first_turn:
            uncached_input_tokens = int(
              getattr(event, "input_tokens", 0) or 0
            )
            cache_creation_input_tokens = int(
              getattr(event, "cache_creation_tokens", 0) or 0
            )
            cache_read_input_tokens = int(
              getattr(event, "cache_read_tokens", 0) or 0
            )
            gross_input_tokens = (
              uncached_input_tokens
              + cache_creation_input_tokens
              + cache_read_input_tokens
            )
            logger.info(
              "[%s] Cache | read=%d create=%d uncached=%d",
              self._sid,
              cache_read_input_tokens,
              cache_creation_input_tokens,
              uncached_input_tokens,
            )
            breakdown = _runner_attr(self, "_token_breakdown_snapshot", _token_breakdown_snapshot)(
              input_tokens=gross_input_tokens,
              system_chars=system_chars,
              tools_chars=tools_chars,
              messages=current_messages,
            )
            if breakdown is not None:
              logger.info(
                "[%s] Token breakdown | system=%d (%d%%) tools=%d (%d%%) messages=%d (%d%%) | total=%d",
                self._sid,
                breakdown.est_system_tokens,
                breakdown.pct_system,
                breakdown.est_tools_tokens,
                breakdown.pct_tools,
                breakdown.est_messages_tokens,
                breakdown.pct_messages,
                breakdown.input_tokens,
                extra={
                  "data": {
                    "event": "token_breakdown",
                    "session_id": self._sid,
                    "turn": turn_count,
                    "input_tokens": breakdown.input_tokens,
                    "est_system_tokens": breakdown.est_system_tokens,
                    "est_tools_tokens": breakdown.est_tools_tokens,
                    "est_messages_tokens": breakdown.est_messages_tokens,
                    "pct_system": breakdown.pct_system,
                    "pct_tools": breakdown.pct_tools,
                    "pct_messages": breakdown.pct_messages,
                  }
                },
              )
          continue

        if event_type == "text_delta":
          text = str(event.text or "")
          if text and result.first_token_t is None:
            result.first_token_t = time_module.time()
            if self._first_text_at is None:
              self._first_text_at = result.first_token_t
          if not suppress_tool_turn_text:
            if text:
              self._append({"type": "text_delta", "text": text})
          result.full_text += text
          continue

        if event_type == "text_end":
          if isinstance(event.raw_block, dict):
            result.content_blocks.append(event.raw_block)
          continue

        if event_type == "thinking_delta":
          thinking_text = str(event.thinking_text or "")
          self._append({"type": "thinking_delta", "text": thinking_text})
          continue

        if event_type == "thinking_end":
          if isinstance(event.raw_block, dict):
            result.content_blocks.append(event.raw_block)
          logger.info("[%s] Thinking block complete | %d chars", self._sid, len(str(event.thinking_text or "")))
          continue

        if event_type == "heartbeat":
          continue

        if event_type == "stream_progress":
          continue

        if event_type == "tool_use_start":
          continue

        if event_type == "tool_use_end":
          if isinstance(event.raw_block, dict):
            history_block = copy.deepcopy(event.raw_block)
            history_block["input"] = copy.deepcopy(
              event.tool_input or {}
            )
            result.content_blocks.append(history_block)
          result.tool_uses.append((event.tool_id, event.tool_name or "tool", dict(event.tool_input or {})))
          continue

        if event_type == "compaction":
          if isinstance(event.raw_block, dict):
            result.content_blocks.append(event.raw_block)
          content = event.raw_block.get("content") if isinstance(event.raw_block, dict) else event.text
          chars = len(content) if isinstance(content, str) else 0
          self._append({"type": "compaction", "chars": chars})
          logger.info("[%s] Compaction block | %d chars", self._sid, chars)
          continue

        if event_type == "usage_update":
          _runner_attr(self, "_apply_usage_update", _apply_usage_update)(
            usage_totals,
            input_tokens=event.input_tokens,
            output_tokens=event.output_tokens,
            reasoning_tokens=event.reasoning_tokens,
            cache_creation_tokens=event.cache_creation_tokens,
            cache_read_tokens=event.cache_read_tokens,
            provider_units=event.provider_units,
            provider_unit_deltas=event.provider_unit_deltas,
          )
          continue

        if event_type == "message_end":
          result.stop_reason = event.stop_reason or None
          result.stop_details = event.stop_details
          result.request_input_tokens = event.request_input_tokens

      logger.debug("[%s] Turn %d stream end", self._sid, turn_count)

    async def _stream_guard(task: asyncio.Task, turn_start_mono: float) -> None:
      nonlocal guard_reason, last_progress_at
      next_progress_log_at = turn_start_mono + STREAM_PROGRESS_LOG_INTERVAL
      while not task.done():
        await asyncio_module.sleep(stream_guard_poll_interval)
        if task.done():
          return
        now = time_module.monotonic()
        if locally_queued(task):
          # Local capacity waiting is not silence from the provider peer.
          # Keep emitting liveness without fabricating peer progress.
          stall = 0.0
        else:
          left_queue_at = local_queue_left_at(task)
          if left_queue_at is not None and left_queue_at > last_progress_at:
            # A queue wait can start and finish entirely between guard polls.
            last_progress_at = left_queue_at
          stall = now - last_progress_at
        if stall > _effective_stall_timeout:
          guard_reason = ("stall", f"no stream progress for {stall:.0f}s")
          logger.error("[%s] Turn %d watchdog (%s): %s", self._sid, turn_count, guard_reason[0], guard_reason[1])
          task.cancel()
          return
        if now >= next_progress_log_at:
          logger.info(
            "[%s] Turn %d streaming: elapsed=%.0fs last_progress=%.0fs ago events=%d",
            self._sid,
            turn_count,
            now - turn_start_mono,
            stall,
            progress_events,
          )
          # Liveness signal: a parent's activity guard stamps a child from it
          # while the child streams output (a large tool input appends nothing
          # else). Clients drop it. Not in the session log; the chat transcript
          # and autonomous events files record it like every streamed event.
          self._append({
            "type": "heartbeat",
            "elapsed_s": int(now - turn_start_mono),
            "last_progress_s": int(stall),
            "events": progress_events,
          })
          next_progress_log_at = now + STREAM_PROGRESS_LOG_INTERVAL

    stream_error: Exception | None = None
    tried_credential_refs = {self._capability_execution.bind.credential_ref}
    transient_retries = 0
    next_retry_delay: float = 0.0
    attempt = -1
    attempt_start_mono = time_module.monotonic()

    def _raise_if_disconnected(exc: Exception | None = None) -> None:
      if not self._disconnected:
        return
      if exc is not None:
        raise exc
      if stream_error is not None:
        raise stream_error
      raise asyncio_module.CancelledError()

    def _retry_delay(next_transient_retry: int) -> float:
      return stream_retry_delay * (stream_retry_backoff ** (next_transient_retry - 1))

    async def _emit_stream_retry(
      attempt: int, attempt_start_mono: float, error: str, *, new_credential: bool = False,
    ) -> None:
      nonlocal next_retry_delay
      if new_credential:
        backoff_s: float | None = 0.0
        next_retry_delay = 0.0
      elif transient_retries < stream_retry_max:
        backoff_s = _retry_delay(transient_retries + 1)
        next_retry_delay = backoff_s
      else:
        backoff_s = None
        next_retry_delay = 0.0
      await self._emit_stream_retry_event(
        attempt=attempt,
        error=error,
        attempt_elapsed_s=time_module.monotonic() - attempt_start_mono,
        backoff_s=backoff_s,
      )

    def _credential_failure_for(exc: Exception) -> ProviderCredentialFailure | None:
      """The credential failure `exc` names, including a typed source error.

      A typed `AnthropicSourceError` (a retired/removed/superseded bound
      generation) is a credential failure exactly like a 401: it goes through
      the same rotation path. Everything else is the provider's own
      classification.
      """
      if isinstance(exc, AnthropicSourceError):
        return ProviderCredentialFailure(
          provider=str(getattr(self._provider, "name", "unknown") or "unknown"),
          kind="auth",
          error_code=getattr(exc, "code", None),
          message=str(exc),
        )
      try:
        return self._provider.classify_credential_failure(exc)
      except Exception as classify_exc:
        logger.warning(
          "[%s] credential failure classification failed (non-fatal): %s",
          self._sid,
          classify_exc,
        )
        return None

    async def _rotate_credential(
      credential_failure: ProviderCredentialFailure,
      attempt: int,
      attempt_start_mono: float,
    ) -> bool:
      """Swap in the provider's next eligible generation; True when it rotated.

      The provider parks the rejected member and the resolver names the
      sibling in a new bind (CRED-ROTATION-REBIND); a generation's first
      attempt does not spend the transient-retry budget, revisiting one does.
      A frozen generation bind never reaches here: it fails typed at
      materialization, before the turn.
      """
      nonlocal transient_retries
      rotated_config = self._provider.next_credential(config, credential_failure)
      if rotated_config is None or not self._apply_refreshed_auth_config(config, rotated_config):
        return False
      credential_ref = self._capability_execution.bind.credential_ref
      new_credential = credential_ref not in tried_credential_refs
      tried_credential_refs.add(credential_ref)
      self._call_metric("gateway.credential_rotated", 1)
      self._append(
        {
          "type": "credential_refreshed",
          "provider": credential_failure.provider,
          "kind": credential_failure.kind,
          "status_code": credential_failure.status_code,
        }
      )
      await _emit_stream_retry(
        attempt,
        attempt_start_mono,
        f"credential rotated after provider {credential_failure.kind} failure",
        new_credential=new_credential,
      )
      if not new_credential:
        transient_retries += 1
      return True

    async def _rotate_after_source_error(
      source_exc: AnthropicSourceError, attempt: int, attempt_start_mono: float,
    ) -> bool:
      failure = _credential_failure_for(source_exc)
      if failure is None:
        return False
      return await _rotate_credential(failure, attempt, attempt_start_mono)

    while transient_retries <= stream_retry_max:
      attempt += 1
      if attempt > 0:
        _raise_if_disconnected()
        await self._close_client(client, timeout=2.0)
        try:
          client = self._provider.create_client(config)
        except AnthropicSourceError as source_exc:
          # The bind's exact generation was retired at its owner between
          # attempts: rotate through the same credential path a 401/429 takes,
          # rather than letting the typed error escape the retry loop.
          if await _rotate_after_source_error(source_exc, attempt, attempt_start_mono):
            continue
          raise
        self._set_client(client)
        logger.warning(
          "[%s] Stream retry %d on turn %d after %s",
          self._sid,
          attempt,
          turn_count,
          _runner_attr(self, "_format_exc", _format_exc)(stream_error) if stream_error is not None else "unknown error",
        )
        _raise_if_disconnected()
        delay = next_retry_delay
        await asyncio_module.sleep(delay)

      last_progress_at = attempt_start_mono = time_module.monotonic()
      progress_events = 0
      commercial_producer = getattr(self, "_commercial_usage_producer", None)
      commercial_guard = getattr(commercial_producer, "assert_work_allowed", None)
      if callable(commercial_guard):
        commercial_guard(self._billing_mode)
      params, advertised_tool_names = _make_params()
      result = _runner_attr(self, "StreamTurnResult", StreamTurnResult)()
      result.advertised_tool_names = advertised_tool_names
      usage_before_attempt = usage_snapshot(usage_totals)
      guard_reason = None
      stream_task = asyncio_module.create_task(_consume_stream(params, result))
      guard_task = asyncio_module.create_task(_stream_guard(stream_task, turn_t0_mono))
      try:
        done, pending = await asyncio_module.wait({stream_task, guard_task}, return_when=first_completed)
        for task in pending:
          task.cancel()
        if pending:
          _, stuck = await asyncio_module.wait(pending, timeout=5.0)
          if stuck:
            logger.warning("[%s] Turn %d: cancelled task stuck, force-closing client", self._sid, turn_count)
            await self._close_client(client, timeout=2.0)
            await asyncio_module.wait(stuck, timeout=2.0)
        if stream_task in done and not stream_task.cancelled():
          exc = stream_task.exception()
          if exc is not None:
            raise exc
      except cancelled_error_type as primary_cancel:
        guard_task.cancel()
        stream_task.cancel()
        primary_error = primary_cancel

        def _record_cancellation_cleanup_failure(
          phase: str,
          cleanup_exc: BaseException,
        ) -> None:
          cleanup_detail = attach_cleanup_failure(
            primary_error,
            cleanup_exc,
          )
          try:
            self._append({
              "type": "run_error",
              "phase": phase,
              "error_type": type(cleanup_exc).__name__,
              "error": cleanup_detail,
              "message": cleanup_detail,
            })
          except Exception:
            pass

        partial_usage_state = _runner_attr(self, "_usage_delta_state", _usage_delta_state)(usage_before_attempt, usage_totals)
        partial_usage = partial_usage_state.usage
        usage_totals.clear()
        usage_totals.update(usage_before_attempt)
        if partial_usage_state.has_tokens:
          try:
            await self._call_on_usage(
              self._build_usage_event(model=config["model"], usage_totals=partial_usage),
              usage_state="canceled",
            )
          except BaseException as cleanup_exc:
            _record_cancellation_cleanup_failure(
              "stream_cancellation_usage",
              cleanup_exc,
            )
        try:
          await self.force_close()
        except BaseException as cleanup_exc:
          _record_cancellation_cleanup_failure(
            "stream_cancellation_cleanup",
            cleanup_exc,
          )
        raise
      except Exception as exc:
        stream_error = exc
        partial_usage_state = _runner_attr(self, "_usage_delta_state", _usage_delta_state)(usage_before_attempt, usage_totals)
        partial_usage = partial_usage_state.usage
        usage_totals.clear()
        usage_totals.update(usage_before_attempt)

        action, guard_error, guard_kind = self._classify_guard_outcome(
          guard_reason,
          transient_retries,
          stream_retry_max,
        )
        if action != "not_guard":
          guard_message = guard_reason[1] if guard_reason else ""
          if action == "retry":
            stream_error = RuntimeError(guard_error)
            if partial_usage_state.has_tokens:
              await self._call_on_usage(
                self._build_usage_event(model=config["model"], usage_totals=partial_usage),
                usage_state="canceled" if self._disconnected else "failed_billable",
              )
            _raise_if_disconnected(stream_error)
            logger.warning(
              "[%s] Stream watchdog stall on turn %d after %.1fs (attempt %d, transient retries %d/%d), retrying: %s",
              self._sid,
              turn_count,
              time_module.time() - turn_t0,
              attempt + 1,
              transient_retries,
              stream_retry_max,
              guard_message,
            )
            await _emit_stream_retry(attempt, attempt_start_mono, guard_error)
            transient_retries += 1
            continue
          logger.error(
            "[%s] Stream watchdog on turn %d after %.1fs (%s): %s | %s",
            self._sid,
            turn_count,
            time_module.time() - turn_t0,
            guard_kind,
            guard_message,
            _runner_attr(self, "_format_exc", _format_exc)(exc),
          )
          if partial_usage_state.has_tokens:
            await self._call_on_usage(
              self._build_usage_event(model=config["model"], usage_totals=partial_usage),
              usage_state="failed_billable",
            )
          return await self._finish_failed_stream_turn(client, guard_error)

        credential_failure = _credential_failure_for(exc)
        if credential_failure is not None:
          # The provider owns interchangeable credentials. A generation's first
          # attempt does not spend the transient-error retry budget; revisiting
          # a generation (including an unchanged bind) does. A typed
          # AnthropicSourceError (a retired bound generation) rotates here too,
          # via the resolver, exactly like a 401.
          usage_totals.clear()
          usage_totals.update(usage_before_attempt)
          if await _rotate_credential(credential_failure, attempt, attempt_start_mono):
            if partial_usage_state.has_tokens:
              await self._call_on_usage(
                self._build_usage_event(model=config["model"], usage_totals=partial_usage),
                usage_state="failed_billable",
              )
            continue
          # No sibling account serves this model: a spent limit, or a denied
          # or uncredited pool. The remaining attempts move to the next chain
          # entry. The failed attempt's partial usage bills the model that
          # served it.
          _exhausted_pool_reason: dict[str, Literal["usage_limit", "credential_denied"]] = {
            "rate_limit": "usage_limit",
            "auth": "credential_denied",
            "billing": "credential_denied",
          }
          hop_reason = _exhausted_pool_reason.get(credential_failure.kind)
          if hop_reason is not None:
            primary_usage = (
              self._build_usage_event(model=config["model"], usage_totals=partial_usage)
              if partial_usage_state.has_tokens
              else None
            )
            if await _hop_model_fallback(hop_reason, credential_failure.reset_at):
              stream_error = RuntimeError(f"model fallback after provider {credential_failure.kind} failure")
              await _emit_stream_retry(
                attempt,
                attempt_start_mono,
                f"model fallback after provider {credential_failure.kind} failure",
                new_credential=True,
              )
              if primary_usage is not None:
                await self._call_on_usage(primary_usage, usage_state="failed_billable")
              # A hop consumes one attempt of the budget but always gets its own.
              transient_retries = min(transient_retries + 1, stream_retry_max)
              continue
            if getattr(self, "_request_boundary_stop_reason", None) == "commercial_budget_reached":
              if primary_usage is not None:
                await self._call_on_usage(primary_usage, usage_state="failed_billable")
              return client, result

        formatted_exc = _runner_attr(self, "_format_exc", _format_exc)(exc)
        logged_exc = _format_error_with_rejection_detail(formatted_exc, exc)
        event_error = error_with_provider_projection(formatted_exc, exc)
        if self._provider.is_context_length_error(exc):
          logger.error(
            "[%s] Stream error on turn %d after %.1fs (context length): %s",
            self._sid,
            turn_count,
            time_module.time() - turn_t0,
            formatted_exc,
          )
          if partial_usage_state.has_tokens:
            await self._call_on_usage(
              self._build_usage_event(model=config["model"], usage_totals=partial_usage),
              usage_state="failed_billable",
            )
          return StreamTurnFailure(
            error=exc,
            formatted_error=formatted_exc,
            is_context_length=True,
          )
        if not self._provider.is_retryable_error(exc):
          logger.error(
            "[%s] Stream error on turn %d after %.1fs (non-retryable): %s",
            self._sid,
            turn_count,
            time_module.time() - turn_t0,
            logged_exc,
          )
          if partial_usage_state.has_tokens:
            await self._call_on_usage(
              self._build_usage_event(model=config["model"], usage_totals=partial_usage),
              usage_state="failed_billable",
            )
          return await self._finish_failed_stream_turn(client, event_error)

        logger.warning(
          "[%s] Transient stream error on turn %d after %.1fs (attempt %d, transient retries %d/%d): %s",
          self._sid,
          turn_count,
          time_module.time() - turn_t0,
          attempt + 1,
          transient_retries,
          stream_retry_max,
          logged_exc,
        )
        if transient_retries < stream_retry_max:
          if partial_usage_state.has_tokens:
            await self._call_on_usage(
              self._build_usage_event(model=config["model"], usage_totals=partial_usage),
              usage_state="canceled" if self._disconnected else "failed_billable",
            )
          _raise_if_disconnected(exc)
          await _emit_stream_retry(attempt, attempt_start_mono, event_error)
          transient_retries += 1
          continue
        if credential_failure is None and not self._disconnected:
          # The provider is still failing after this request's retry budget:
          # the next chain entry gets the remaining attempt. A disconnected
          # client takes today's path below, which bills the partial usage.
          primary_usage = (
            self._build_usage_event(model=config["model"], usage_totals=partial_usage)
            if partial_usage_state.has_tokens
            else None
          )
          if await _hop_model_fallback("provider_outage", None):
            stream_error = exc
            await _emit_stream_retry(attempt, attempt_start_mono, event_error, new_credential=True)
            if primary_usage is not None:
              await self._call_on_usage(primary_usage, usage_state="failed_billable")
            continue
          if getattr(self, "_request_boundary_stop_reason", None) == "commercial_budget_reached":
            if primary_usage is not None:
              await self._call_on_usage(primary_usage, usage_state="failed_billable")
            return client, result
        if partial_usage_state.has_tokens:
          await self._call_on_usage(
            self._build_usage_event(model=config["model"], usage_totals=partial_usage),
            usage_state="failed_billable",
          )
        transient_retries += 1
      else:
        action, guard_error, guard_kind = self._classify_guard_outcome(
          guard_reason,
          transient_retries,
          stream_retry_max,
        )
        if action != "not_guard":
          guard_message = guard_reason[1] if guard_reason else ""
          partial_usage_state = _runner_attr(self, "_usage_delta_state", _usage_delta_state)(usage_before_attempt, usage_totals)
          partial_usage = partial_usage_state.usage
          usage_totals.clear()
          usage_totals.update(usage_before_attempt)
          if action == "retry":
            stream_error = RuntimeError(guard_error)
            if partial_usage_state.has_tokens:
              await self._call_on_usage(
                self._build_usage_event(model=config["model"], usage_totals=partial_usage),
                usage_state="canceled" if self._disconnected else "failed_billable",
              )
            _raise_if_disconnected(stream_error)
            logger.warning(
              "[%s] Stream watchdog stall on turn %d after %.1fs (attempt %d, transient retries %d/%d), retrying: %s",
              self._sid,
              turn_count,
              time_module.time() - turn_t0,
              attempt + 1,
              transient_retries,
              stream_retry_max,
              guard_message,
            )
            await _emit_stream_retry(attempt, attempt_start_mono, guard_error)
            transient_retries += 1
            continue
          if guard_kind == "stall" and not self._disconnected:
            # A provider that stays silent through the retry budget is failing
            # exactly like one that errors through it (design §R9 outage
            # trigger), so the last stall takes the exception branch's hop.
            primary_usage = (
              self._build_usage_event(model=config["model"], usage_totals=partial_usage)
              if partial_usage_state.has_tokens
              else None
            )
            if await _hop_model_fallback("provider_outage", None):
              stream_error = RuntimeError(guard_error)
              await _emit_stream_retry(attempt, attempt_start_mono, guard_error, new_credential=True)
              if primary_usage is not None:
                await self._call_on_usage(primary_usage, usage_state="failed_billable")
              continue
            if getattr(self, "_request_boundary_stop_reason", None) == "commercial_budget_reached":
              if primary_usage is not None:
                await self._call_on_usage(primary_usage, usage_state="failed_billable")
              return client, result
          if partial_usage_state.has_tokens:
            await self._call_on_usage(
              self._build_usage_event(model=config["model"], usage_totals=partial_usage),
              usage_state="failed_billable",
            )
          return await self._finish_failed_stream_turn(client, guard_error)
        if suppress_tool_turn_text:
          if result.tool_uses:
            if result.full_text:
              logger.info(
                "[%s] Suppressed %d chars of assistant text from tool-use turn %d",
                self._sid,
                len(result.full_text),
                turn_count,
              )
            result.full_text = ""
            result.content_blocks = [
              block
              for block in result.content_blocks
              if not (isinstance(block, dict) and block.get("type") == "text")
            ]
          elif result.full_text:
            self._append({"type": "text_delta", "text": result.full_text})
        return client, result

    if stream_error is not None:
      formatted_exc = _runner_attr(self, "_format_exc", _format_exc)(stream_error)
      logged_exc = _format_error_with_rejection_detail(formatted_exc, stream_error)
      event_error = error_with_provider_projection(formatted_exc, stream_error)
      if self._provider.is_context_length_error(stream_error):
        logger.error(
          "[%s] Stream failed on turn %d after %d retries (context length): %s",
          self._sid,
          turn_count,
          stream_retry_max,
          formatted_exc,
        )
        return StreamTurnFailure(
          error=stream_error,
          formatted_error=formatted_exc,
          is_context_length=True,
        )
      logger.error(
        "[%s] Stream failed on turn %d after %d retries: %s",
        self._sid,
        turn_count,
        stream_retry_max,
        logged_exc,
      )
      return await self._finish_failed_stream_turn(client, event_error)
    return None
