from __future__ import annotations

import asyncio
import copy
import logging
import sys
import time
from typing import Any, Dict, List, Optional, Tuple, Union

from .auth import ProviderCredentialFailure
from .capability_binding import validate_reported_identity
from .providers import ModelInfo, ThinkingLevel
from .thinking import EffortResolution, parse_effort
from .runner_introspection import format_exc as _format_exc
from .runner_limits import (
  effective_compaction_trigger as _effective_compaction_trigger,
  token_breakdown_snapshot as _token_breakdown_snapshot,
)
from .runner_prompt_rules import (
  messages_require_tool_only_turns as _messages_require_tool_only_turns,
  system_prompt_requires_tool_only_turns as _system_prompt_requires_tool_only_turns,
)
from .runner_cleanup import attach_cleanup_failure
from .runner_session_lifecycle import _runner_attr
from .runner_state import StreamTurnFailure, StreamTurnResult
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


log = logging.getLogger("agent_gateway.runner")
STREAM_GUARD_POLL_INTERVAL = 2.0
STREAM_RETRY_MAX = 3
STREAM_RETRY_DELAY = 2.0
STREAM_RETRY_BACKOFF = 2.0


def _runner_module_attr(name: str, fallback: Any) -> Any:
  module = sys.modules.get("agent_gateway.runner")
  if module is None:
    return fallback
  return getattr(module, name, fallback)


class RunnerStreamTurnMixin:
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
    usage_totals: Dict[str, Any],
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
    guard_reason: tuple[str, str] | None = None
    requested_effort = parse_effort(
      config.get("effort"),
      field_name="capability_execution.effort",
    )
    if requested_effort is None:
      raise ValueError(
        "native execution requires an explicitly bound effort"
      )
    bound_effort = self._capability_execution.bind.effort
    if requested_effort.value != bound_effort:
      raise ValueError(
        "runtime effort does not match the immutable capability bind"
      )
    effort_resolution = self._provider.resolve_effort(
      requested=requested_effort,
      model=config["model"],
      model_info=model_info,
      max_tokens=max_tokens,
      auth_mode=config.get("auth_mode"),
      base_url=config.get("base_url") or config.get("baseURL"),
      compat=config.get("compat"),
    )
    if (
      effort_resolution.requested != requested_effort
      or effort_resolution.effective != requested_effort
    ):
      raise ValueError(
        (
          f"native execution cannot preserve bound effort "
          f"{requested_effort.value!r} for "
          f"{self._capability_execution.bind.provider}:"
          f"{self._capability_execution.bind.upstream_model} "
          f"with max_tokens={max_tokens}"
        )
      )
    self._effort_resolution = effort_resolution
    _effective_stall_timeout = self._effective_stream_stall_timeout(
      config=config,
      model_info=model_info,
      max_tokens=max_tokens,
      current_messages=current_messages,
      effort_resolution=effort_resolution,
    )

    def _make_params() -> tuple[Dict[str, Any], frozenset[str]]:
      normalized_messages = self._provider.normalize_messages(current_messages, model_info)
      request_tools = copy.deepcopy(base_kwargs.get("tools") or [])
      advertised_tool_names = frozenset(
        str(definition.get("name") or "").strip()
        for definition in request_tools
        if isinstance(definition, dict)
        and str(definition.get("name") or "").strip()
      )
      fork_kwargs: Dict[str, Any] = {}
      if getattr(self._provider, "name", None) == "anthropic":
        fork_kwargs = {
          "fork_mode": bool(getattr(self, "_fork_mode", False)),
          "fork_marker_position": getattr(
            self,
            "_fork_marker_position",
            None,
          ),
        }
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
        ),
        compaction_instructions=self._compaction_instructions,
        **fork_kwargs,
      )
      rendered_system_blocks: list[tuple[str, bool]] = []
      for block in params.get("system") or []:
        if not isinstance(block, dict):
          continue
        text = str(block.get("text") or "")
        if text:
          rendered_system_blocks.append(
            (text, "cache_control" in block)
          )
      marker_locations = [
        (message_index, block_index)
        for message_index, message in enumerate(params.get("messages") or [])
        for block_index, block in enumerate(
          message.get("content")
          if isinstance(message, dict)
          and isinstance(message.get("content"), list)
          else []
        )
        if isinstance(block, dict) and "cache_control" in block
      ]
      self._last_request_system_blocks = tuple(
        rendered_system_blocks
      )
      self._last_request_wire_tools = copy.deepcopy(
        params.get("tools") or []
      )
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
      nonlocal last_progress_at
      first_turn = turn_count == 1
      logger.debug("[%s] Turn %d stream open", self._sid, turn_count)

      async for event in self._provider.stream(client, params):
        event_type = event.type
        if event_type != "heartbeat":
          last_progress_at = time_module.monotonic()

        if event_type == "message_start":
          bind = self._capability_execution.bind
          usage_totals.update({
            "capability_bind": bind.receipt(),
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
          )
          if first_turn:
            logger.info(
              "[%s] Cache | read=%d create=%d uncached=%d",
              self._sid,
              event.cache_read_tokens,
              event.cache_creation_tokens,
              event.input_tokens,
            )
            breakdown = _runner_attr(self, "_token_breakdown_snapshot", _token_breakdown_snapshot)(
              input_tokens=event.input_tokens,
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
          if result.first_token_t is None:
            result.first_token_t = time_module.time()
          text = str(event.text or "")
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
            result.content_blocks.append(event.raw_block)
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

      logger.debug("[%s] Turn %d stream end", self._sid, turn_count)

    async def _stream_guard(task: asyncio.Task, turn_start_mono: float) -> None:
      nonlocal guard_reason
      while not task.done():
        await asyncio_module.sleep(stream_guard_poll_interval)
        if task.done():
          return
        now = time_module.monotonic()
        stall = now - last_progress_at
        if stall > _effective_stall_timeout:
          guard_reason = ("stall", f"no stream progress for {stall:.0f}s")
          logger.error("[%s] Turn %d watchdog (%s): %s", self._sid, turn_count, guard_reason[0], guard_reason[1])
          task.cancel()
          return
        if self._per_turn_timeout is not None and (now - turn_start_mono) > self._per_turn_timeout:
          guard_reason = ("timeout", f"turn timeout after {now - turn_start_mono:.0f}s")
          logger.error("[%s] Turn %d watchdog (%s): %s", self._sid, turn_count, guard_reason[0], guard_reason[1])
          task.cancel()
          return

    stream_error: Exception | None = None

    def _raise_if_disconnected(exc: Exception | None = None) -> None:
      if not self._disconnected:
        return
      if exc is not None:
        raise exc
      if stream_error is not None:
        raise stream_error
      raise asyncio_module.CancelledError()

    credential_refresh_attempted = False

    for attempt in range(1 + stream_retry_max):
      if attempt > 0:
        _raise_if_disconnected()
        await self._close_client(client, timeout=2.0)
        client = self._provider.create_client(config, timeout=self._client_timeout)
        self._set_client(client)
        logger.warning(
          "[%s] Stream retry %d/%d on turn %d after %s",
          self._sid,
          attempt,
          stream_retry_max,
          turn_count,
          _runner_attr(self, "_format_exc", _format_exc)(stream_error) if stream_error is not None else "unknown error",
        )
        _raise_if_disconnected()
        delay = stream_retry_delay * (stream_retry_backoff ** (attempt - 1))
        await asyncio_module.sleep(delay)

      last_progress_at = time_module.monotonic()
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
          attempt,
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
              "[%s] Stream watchdog stall on turn %d after %.1fs (attempt %d/%d), retrying: %s",
              self._sid,
              turn_count,
              time_module.time() - turn_t0,
              attempt + 1,
              1 + stream_retry_max,
              guard_message,
            )
            await self._emit_stream_retry_event(attempt=attempt, error=guard_error)
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
          await self._emit_error_event(guard_error)
          await self._close_client(client, timeout=5.0)
          return None

        credential_failure: ProviderCredentialFailure | None = None
        try:
          credential_failure = self._provider.classify_credential_failure(exc)
        except Exception as classify_exc:
          logger.warning("[%s] credential failure classification failed (non-fatal): %s", self._sid, classify_exc)
        if credential_failure is not None and not credential_refresh_attempted:
          credential_refresh_attempted = True
          refreshed_config = await self._call_credential_refresher(credential_failure)
          if refreshed_config is not None:
            self._apply_refreshed_auth_config(config, refreshed_config)
            usage_totals.clear()
            usage_totals.update(usage_before_attempt)
            stream_error = RuntimeError(f"credential refreshed after {credential_failure.kind}")
            self._call_metric("gateway.credential_refresh_success", 1)
            self._append(
              {
                "type": "credential_refreshed",
                "provider": credential_failure.provider,
                "kind": credential_failure.kind,
                "status_code": credential_failure.status_code,
              }
            )
            await self._emit_stream_retry_event(
              attempt=attempt,
              error=f"credential refreshed after provider {credential_failure.kind} failure",
            )
            if partial_usage_state.has_tokens:
              await self._call_on_usage(
                self._build_usage_event(model=config["model"], usage_totals=partial_usage),
                usage_state="failed_billable",
              )
            continue

        formatted_exc = _runner_attr(self, "_format_exc", _format_exc)(exc)
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
            formatted_exc,
          )
          if partial_usage_state.has_tokens:
            await self._call_on_usage(
              self._build_usage_event(model=config["model"], usage_totals=partial_usage),
              usage_state="failed_billable",
            )
          await self._emit_error_event(formatted_exc)
          await self._close_client(client, timeout=5.0)
          return None

        logger.warning(
          "[%s] Transient stream error on turn %d after %.1fs (attempt %d/%d): %s",
          self._sid,
          turn_count,
          time_module.time() - turn_t0,
          attempt + 1,
          1 + stream_retry_max,
          formatted_exc,
        )
        if attempt < stream_retry_max:
          if partial_usage_state.has_tokens:
            await self._call_on_usage(
              self._build_usage_event(model=config["model"], usage_totals=partial_usage),
              usage_state="canceled" if self._disconnected else "failed_billable",
            )
          _raise_if_disconnected(exc)
          await self._emit_stream_retry_event(attempt=attempt, error=formatted_exc)
          continue
        if partial_usage_state.has_tokens:
          await self._call_on_usage(
            self._build_usage_event(model=config["model"], usage_totals=partial_usage),
            usage_state="failed_billable",
          )
      else:
        action, guard_error, _ = self._classify_guard_outcome(
          guard_reason,
          attempt,
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
              "[%s] Stream watchdog stall on turn %d after %.1fs (attempt %d/%d), retrying: %s",
              self._sid,
              turn_count,
              time_module.time() - turn_t0,
              attempt + 1,
              1 + stream_retry_max,
              guard_message,
            )
            await self._emit_stream_retry_event(attempt=attempt, error=guard_error)
            continue
          if partial_usage_state.has_tokens:
            await self._call_on_usage(
              self._build_usage_event(model=config["model"], usage_totals=partial_usage),
              usage_state="failed_billable",
            )
          await self._emit_error_event(guard_error)
          await self._close_client(client, timeout=5.0)
          return None
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
        formatted_exc,
      )
      await self._emit_error_event(formatted_exc)
      await self._close_client(client, timeout=5.0)
    return None
