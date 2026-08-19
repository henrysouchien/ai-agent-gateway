from __future__ import annotations

import asyncio
from fnmatch import fnmatchcase
import json
import logging
import os
import time
import uuid
from dataclasses import replace
from typing import Any, AsyncIterator, Awaitable, Callable, Dict, List, Sequence

from .approval_policy import (
  ApprovalDecision as PolicyApprovalDecision,
  ApprovalPolicy,
  ApprovalRequest as PolicyApprovalRequest,
  RunContext,
  apply_decision_to_request,
  build_approval_request,
  call_policy_safely,
  sha256_args,
  utc_now,
)
from . import approval_settings
from .approval_enrichment import effective_trade_approval_expiry_seconds, enrich_trade_approval_args
from .capability_execution import BoundCapabilityExecution
from .event_log import EventLog
from .context_capture import ContextCapture, build_context_manifest_event, canonical_manifest_digest
from .multi_user.billing import SessionUsageSummary, UsageEvent, _UsageAggregator, normalize_identity
from .policy_imports import resolve_server_policy_tool_class
from .product_config import gateway_product_id
from .providers.agent_sdk import AgentSDKConfig, estimate_cost, _validate_sdk_version
from .providers.anthropic import _server_tool_unit_deltas
from .usage_resilience import CommercialUsageCircuitOpen
from .runner import (
  ToolResultContext,
  _ACTIVE_SKILL_ALLOW_RESULT_KEY,
  _ACTIVE_SKILL_DENY_RESULT_KEY,
  _ACTIVE_SKILL_REPORT_DOORS_RESULT_KEY,
  _REPORT_DOOR_CLEAR_SUCCESS_STATUSES,
  _detect_user_id_param,
)
from .runner_introspection import detect_keyword_param as _detect_keyword_param
from .run_identity import (
  MODEL_RUN_IDENTITY_LOCAL_TOOLS,
  RunIdentityCarrier,
  RunIdentityCarrierError,
  inject_run_identity_into_mcp_server_configs,
  validate_run_identity,
)
from .runner_skill_gate import is_report_door_clear_event as _is_report_door_clear_event
from .runner_session_events import build_user_message_event
from .selected_content import (
  SelectedContentBinding,
  serialize_selected_content_bindings,
)
from .session_recap import emit_recap_then_terminal
from .secret_boundary import SecretBoundary, sanitize_boundary_value, sanitize_tool_event
from .skill_context import clear_current_skill, current_skill
from .workflow_evidence_provenance import (
  WORKFLOW_EVIDENCE_PROJECTION_RESULT_KEY as _WORKFLOW_EVIDENCE_PROJECTION_RESULT_KEY,
)
from . import sdk_runner_approval as _sdk_runner_approval
from . import sdk_runner_context as _sdk_runner_context
from . import sdk_runner_helpers as _sdk_runner_helpers
from . import sdk_runner_stream as _sdk_runner_stream


log = logging.getLogger("agent_gateway.sdk_runner")

OnToolResult = Callable[[ToolResultContext], Awaitable[List[Dict[str, Any]] | None]]
OnUsage = Callable[[UsageEvent], Awaitable[None] | None]
OnSessionSummary = Callable[[SessionUsageSummary], Awaitable[None] | None]
OnToolTiming = Callable[..., None]


_PATCH_OP_RAW_INPUT_TOOLS = _sdk_runner_helpers.PATCH_OP_RAW_INPUT_TOOLS
_TRUSTED_SDK_LOAD_TOOLS_ID = "mcp__gateway-tools__load_tools"
_as_dict = _sdk_runner_helpers.as_dict
_as_plain_dict = _sdk_runner_helpers.as_plain_dict
_extract_text = _sdk_runner_helpers.extract_text
_get_attr = _sdk_runner_helpers.get_attr
_join_system_prompt = _sdk_runner_helpers.join_system_prompt
_parse_result_payload = _sdk_runner_helpers.parse_result_payload
_policy_owner_mismatch = _sdk_runner_helpers.policy_owner_mismatch
_policy_tool_name = _sdk_runner_helpers.policy_tool_name
_redact_tool_input_for_event = _sdk_runner_helpers.redact_tool_input_for_event
_server_for_tool = _sdk_runner_helpers.server_for_tool
_should_escrow_raw_tool_input = _sdk_runner_helpers.should_escrow_raw_tool_input
_summarize_error_payload = _sdk_runner_helpers.summarize_error_payload


def _approval_queue_timeout_seconds(expiry_seconds: float | int | None) -> float:
  return _sdk_runner_approval.approval_queue_timeout_seconds(
    expiry_seconds,
    approval_wait_seconds_fn=approval_settings.approval_wait_seconds,
  )


def _agent_sdk_credential_env(
  capability_execution: BoundCapabilityExecution,
) -> dict[str, str]:
  auth_config = capability_execution.auth_config
  auth_mode = str(auth_config.get("auth_mode") or "").strip().lower()
  if auth_mode not in {"api", "oauth"}:
    raise RuntimeError(
      "AgentSDKRunner capability execution requires explicit api or oauth auth_mode"
    )
  api_key = str(auth_config.get("api_key") or "").strip()
  auth_token = str(auth_config.get("auth_token") or "").strip()
  if auth_mode == "api" and not api_key:
    raise RuntimeError(
      "AgentSDKRunner API capability execution requires api_key"
    )
  if auth_mode == "oauth" and not auth_token:
    raise RuntimeError(
      "AgentSDKRunner OAuth capability execution requires auth_token"
    )
  return {
    "ANTHROPIC_API_KEY": "" if auth_mode == "oauth" else api_key,
    "ANTHROPIC_AUTH_TOKEN": auth_token if auth_mode == "oauth" else "",
    # ClaudeAgentOptions.env is merged with the parent process environment.
    # Explicitly clear alternate credential and routing selectors so the
    # child cannot escape the immutable session.driver credential/route.
    "CLAUDE_CODE_OAUTH_TOKEN": "",
    "CLAUDE_CODE_USE_BEDROCK": "",
    "CLAUDE_CODE_USE_VERTEX": "",
    "CLAUDE_CODE_USE_FOUNDRY": "",
    "ANTHROPIC_BASE_URL": "",
  }


class _PromptMessages:
  def __init__(self, text: str) -> None:
    self.text = text

  def __contains__(self, needle: object) -> bool:
    return isinstance(needle, str) and needle in self.text

  def __str__(self) -> str:
    return self.text

  async def __aiter__(self) -> AsyncIterator[dict[str, Any]]:
    yield {
      "type": "user",
      "message": {
        "role": "user",
        "content": [{"type": "text", "text": self.text}],
      },
    }


class _SDKToolAdmissionSet(set[str]):
  """Set-like view that applies SDK wildcard and MCP-server admission rules."""

  def __init__(self, values: Sequence[str], runner: "AgentSDKRunner") -> None:
    super().__init__(values)
    self._runner = runner

  def __contains__(self, value: object) -> bool:
    if not isinstance(value, str):
      return super().__contains__(value)
    if super().__contains__(value):
      return True
    if any("*" in pattern and fnmatchcase(value, pattern) for pattern in self):
      return True
    if value.startswith("mcp__") and self._runner._sdk_admission_enforced:
      parts = value.split("__", 2)
      if len(parts) != 3 or not parts[1] or not parts[2]:
        return True
      server_name = parts[1]
      if server_name not in self._runner._mcp_server_configs:
        return True
      advertised = self._runner._sdk_advertised_mcp_tool_ids_by_server.get(
        server_name
      )
      if advertised is None:
        return True
      return value not in advertised
    return False


class _SDKResultProxy:
  def __init__(
    self,
    message: Any,
    *,
    usage: dict[str, Any],
    num_turns: int,
    total_cost_usd: float | None,
  ) -> None:
    self._message = message
    self.usage = usage
    self.num_turns = num_turns
    self.total_cost_usd = total_cost_usd

  def __getattr__(self, name: str) -> Any:
    return getattr(self._message, name)


class _SDKSegmentQueryIterator:
  """One run-scoped iterator over one or more SDK query segments."""

  def __init__(
    self,
    *,
    runner: "AgentSDKRunner",
    sdk_module: Any,
    initial_prompt: str,
    options_kwargs: dict[str, Any],
  ) -> None:
    self._runner = runner
    self._sdk = sdk_module
    self._initial_prompt = initial_prompt
    self._options_kwargs = options_kwargs
    self._iterator: Any = None
    self._closed = False
    self._prior_usage = {
      "input_tokens": 0,
      "output_tokens": 0,
      "cache_read_input_tokens": 0,
      "cache_creation_input_tokens": 0,
      "provider_unit_deltas": {},
    }
    self._prior_turns = 0
    self._prior_cost_usd = 0.0
    self._segment_usage = {
      "input_tokens": 0,
      "output_tokens": 0,
      "cache_read_input_tokens": 0,
      "cache_creation_input_tokens": 0,
      "provider_unit_deltas": {},
    }
    self._segment_turns = 0
    self._current_call_usage: dict[str, Any] | None = None
    self._resume_session_id: str | None = None
    self._run_max_turns = options_kwargs.get("max_turns")
    self._run_max_budget = options_kwargs.get("max_budget_usd")
    self._accounting_projected = False

  def __aiter__(self) -> "_SDKSegmentQueryIterator":
    return self

  @staticmethod
  def _merge_usage(target: dict[str, Any], source: dict[str, Any]) -> None:
    for key in (
      "input_tokens",
      "output_tokens",
      "cache_read_input_tokens",
      "cache_creation_input_tokens",
    ):
      target[key] = int(target.get(key) or 0) + int(source.get(key) or 0)
    target_units = dict(target.get("provider_unit_deltas") or {})
    source_units = dict(source.get("provider_unit_deltas") or {})
    if not source_units:
      source_units = _server_tool_unit_deltas(source)
    for unit_name, count in source_units.items():
      target_units[str(unit_name)] = int(target_units.get(str(unit_name)) or 0) + int(count or 0)
    target["provider_unit_deltas"] = target_units

  @staticmethod
  def _with_raw_provider_units(usage: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(usage)
    units = dict(normalized.get("provider_unit_deltas") or {})
    raw_units: dict[str, int] = {}
    for unit_name, count in units.items():
      if unit_name == "web_search":
        raw_units["web_search_requests"] = int(count or 0)
      elif unit_name == "web_fetch":
        raw_units["web_fetch_requests"] = int(count or 0)
    if raw_units:
      normalized["server_tool_use"] = raw_units
    return normalized

  async def _project_interrupted_accounting(self) -> None:
    if self._accounting_projected:
      return
    self._accounting_projected = True
    self._runner._update_usage(
      self._with_raw_provider_units(self._prior_usage),
      total_cost_usd=self._prior_cost_usd,
      num_turns=self._prior_turns,
    )
    await self._runner._emit_usage_hook(
      usage_state="failed_billable",
      emit_commercial=not self._runner._commercial_usage_emitted,
    )

  def _observe_message(self, message: Any) -> None:
    session_id = str(_get_attr(message, "session_id") or "").strip()
    if not session_id and hasattr(message, "data"):
      session_id = str(
        _as_dict(_get_attr(message, "data")).get("session_id") or ""
      ).strip()
    if session_id:
      self._resume_session_id = session_id
      self._runner._sdk_resume_session_id = session_id
    if hasattr(message, "event"):
      raw_event = _as_dict(getattr(message, "event"))
      event_type = str(raw_event.get("type") or "")
      if event_type == "message_start":
        message_payload = _as_dict(raw_event.get("message"))
        usage = _as_dict(message_payload.get("usage"))
        self._segment_turns += 1
        self._current_call_usage = {
          "input_tokens": int(usage.get("input_tokens") or 0),
          "output_tokens": 0,
          "cache_read_input_tokens": int(usage.get("cache_read_input_tokens") or 0),
          "cache_creation_input_tokens": int(usage.get("cache_creation_input_tokens") or 0),
          "provider_unit_deltas": _server_tool_unit_deltas(usage),
        }
      elif event_type == "message_delta":
        usage = _as_dict(raw_event.get("usage"))
        current = dict(self._current_call_usage or {})
        current["output_tokens"] = int(usage.get("output_tokens") or 0)
        current["provider_unit_deltas"] = (
          _server_tool_unit_deltas(usage)
          or dict(current.get("provider_unit_deltas") or {})
        )
        self._merge_usage(self._segment_usage, current)
        self._current_call_usage = None
      return
    if hasattr(message, "model") and hasattr(message, "content"):
      if self._segment_turns == 0:
        self._segment_turns = 1

  def _finalize_interrupted_segment(self, result_message: Any | None = None) -> None:
    if self._current_call_usage is not None:
      self._merge_usage(self._segment_usage, self._current_call_usage)
      self._current_call_usage = None
    segment_usage = self._segment_usage
    observed_turns = self._segment_turns or 1
    exact_cost: float | None = None
    if result_message is not None:
      result_usage = _as_dict(_get_attr(result_message, "usage"))
      if result_usage:
        result_provider_units = dict(
          result_usage.get("provider_unit_deltas") or {}
        ) or _server_tool_unit_deltas(result_usage)
        segment_usage = {
          "input_tokens": int(result_usage.get("input_tokens") or 0),
          "output_tokens": int(result_usage.get("output_tokens") or 0),
          "cache_read_input_tokens": int(
            result_usage.get("cache_read_input_tokens") or 0
          ),
          "cache_creation_input_tokens": int(
            result_usage.get("cache_creation_input_tokens") or 0
          ),
          "provider_unit_deltas": result_provider_units,
        }
      observed_turns = int(_get_attr(result_message, "num_turns") or 0) or observed_turns
      raw_cost = _get_attr(result_message, "total_cost_usd")
      if isinstance(raw_cost, (int, float)):
        exact_cost = float(raw_cost)
    self._merge_usage(self._prior_usage, segment_usage)
    self._prior_turns += observed_turns
    if exact_cost is None:
      segment_cost = estimate_cost(
        self._runner._effective_model,
        int(segment_usage.get("input_tokens") or 0),
        int(segment_usage.get("output_tokens") or 0),
        cache_read_tokens=int(segment_usage.get("cache_read_input_tokens") or 0),
        cache_creation_tokens=int(
          segment_usage.get("cache_creation_input_tokens") or 0
        ),
      )
      exact_cost = float(segment_cost.total or 0.0)
    self._prior_cost_usd += exact_cost
    self._segment_usage = {
      "input_tokens": 0,
      "output_tokens": 0,
      "cache_read_input_tokens": 0,
      "cache_creation_input_tokens": 0,
      "provider_unit_deltas": {},
    }
    self._segment_turns = 0

  async def _close_current(self) -> None:
    iterator = self._iterator
    self._iterator = None
    if iterator is None:
      return
    close_fn = getattr(iterator, "aclose", None) or getattr(iterator, "close", None)
    if close_fn is None:
      return
    try:
      closed = close_fn()
      if asyncio.iscoroutine(closed):
        await closed
    except Exception:
      pass

  def _remaining_limits(self) -> tuple[int | None, float | None]:
    remaining_turns = None
    if isinstance(self._run_max_turns, int):
      remaining_turns = self._run_max_turns - self._prior_turns
    remaining_budget = None
    if isinstance(self._run_max_budget, (int, float)):
      remaining_budget = float(self._run_max_budget) - self._prior_cost_usd
    return remaining_turns, remaining_budget

  async def _rebuild(self, result_message: Any | None = None) -> None:
    self._finalize_interrupted_segment(result_message)
    try:
      if not self._resume_session_id:
        self._resume_session_id = self._runner._sdk_resume_session_id
      if not self._resume_session_id:
        raise RuntimeError(
          "sdk_tool_rebuild_transcript_unavailable: SDK session identity was not observed"
        )
      remaining_turns, remaining_budget = self._remaining_limits()
      if remaining_turns is not None and remaining_turns <= 0:
        raise RuntimeError(
          "sdk_tool_rebuild_limits_exhausted: no run-scoped turns remain"
        )
      if remaining_budget is not None and remaining_budget <= 0:
        raise RuntimeError(
          "sdk_tool_rebuild_limits_exhausted: no run-scoped budget remains"
        )
      pending_error = self._runner._pending_sdk_load_error(
        self._runner._pending_sdk_load
      )
      if pending_error is not None:
        raise RuntimeError(
          f"{pending_error['code']}: {pending_error['message']}"
        )
      changed, error = self._runner._apply_pending_sdk_load()
      if error is not None or not changed:
        raise RuntimeError(error or "sdk_tool_rebuild_failed")
    except Exception:
      self._runner._discard_pending_sdk_load()
      await self._project_interrupted_accounting()
      raise
    await self._close_current()
    self._options_kwargs["mcp_servers"] = dict(self._runner._mcp_server_configs)
    self._options_kwargs["disallowed_tools"] = list(self._runner._disallowed_tools)
    self._options_kwargs["resume"] = self._resume_session_id
    self._options_kwargs["continue_conversation"] = False
    if remaining_turns is not None:
      self._options_kwargs["max_turns"] = remaining_turns
    if remaining_budget is not None:
      self._options_kwargs["max_budget_usd"] = remaining_budget

  def _continuation_prompt(self) -> str:
    load_results = "\n".join(self._runner._sdk_rebuild_transcript)
    return (
      "The SDK query was rebuilt after load_tools changed the advertised MCP tool set. "
      "Continue the resumed run without repeating completed work. The load result was:\n\n"
      f"{load_results}"
    )

  def _start_segment(self) -> None:
    prompt_text = (
      self._initial_prompt
      if self._iterator is None and self._prior_turns == 0
      else self._continuation_prompt()
    )
    options = getattr(self._sdk, "ClaudeAgentOptions")(**self._options_kwargs)
    self._iterator = getattr(self._sdk, "query")(
      prompt=_PromptMessages(prompt_text),
      options=options,
    )

  def _aggregate_result(self, message: Any) -> Any:
    result_usage = _as_dict(_get_attr(message, "usage"))
    aggregate = dict(self._prior_usage)
    self._merge_usage(aggregate, result_usage)
    aggregate = self._with_raw_provider_units(aggregate)
    result_turns = int(_get_attr(message, "num_turns") or 0)
    result_cost = _get_attr(message, "total_cost_usd")
    total_cost = self._prior_cost_usd + float(result_cost or 0.0)
    self._options_kwargs["max_turns"] = self._run_max_turns
    self._options_kwargs["max_budget_usd"] = self._run_max_budget
    return _SDKResultProxy(
      message,
      usage=aggregate,
      num_turns=self._prior_turns + result_turns,
      total_cost_usd=total_cost if result_cost is not None or self._prior_cost_usd else None,
    )

  async def __anext__(self) -> Any:
    if self._closed:
      raise StopAsyncIteration
    while True:
      if self._runner._pending_sdk_load and self._iterator is None:
        await self._rebuild()
      if self._iterator is None:
        self._start_segment()
      try:
        message = await self._iterator.__anext__()
      except StopAsyncIteration:
        if self._runner._pending_sdk_load:
          await self._rebuild()
          continue
        raise
      self._observe_message(message)
      if self._runner._pending_sdk_load:
        if hasattr(message, "duration_ms") and hasattr(message, "num_turns"):
          await self._rebuild(message)
          continue
        return message
      if hasattr(message, "duration_ms") and hasattr(message, "num_turns"):
        return self._aggregate_result(message)
      return message

  async def aclose(self) -> None:
    self._closed = True
    self._runner._discard_pending_sdk_load()
    await self._close_current()


class AgentSDKRunner(_sdk_runner_stream._SDKRunnerStreamMixin):
  """Run a conversation through the Anthropic agent SDK.

  This is an alternative to `AgentRunner` when you want to delegate tool-loop
  execution to the pinned SDK while keeping the same gateway HTTP surface.
  """

  def __init__(
    self,
    event_log: EventLog,
    session_id: str,
    *,
    sdk_config: AgentSDKConfig,
    capability_execution: BoundCapabilityExecution,
    system_prompt: str,
    disallowed_tools: list[str] | None = None,
    mcp_server_configs: dict | None = None,
    allowed_tools: list[str] | None = None,
    max_turns: int | None = None,
    max_tokens_override: int | None = None,
    on_usage: Callable[..., Any] | None = None,
    on_session_summary: Callable[..., Any] | None = None,
    on_late_usage_event: Callable[..., Any] | None = None,
    on_tool_result: Callable[..., Any] | None = None,
    on_tool_timing: Callable[..., Any] | None = None,
    provider_id_for_tool: Callable[[str], str | None] | None = None,
    _parent_aggregator: _UsageAggregator | None = None,
    session: Any | None = None,
    store: Any | None = None,
    policy: ApprovalPolicy | None = None,
    run_context: RunContext | None = None,
    skill_run_id: str | None = None,
    mcp_run_identity_servers: frozenset[str] | None = None,
    workspace_dir: str | None = None,
    batch_id: int | str | None = None,
    started_at: float | None = None,
    emit_session_recap: bool = True,
    context_surfaces: list[dict[str, Any]] | Callable[[], list[dict[str, Any]]] | None = None,
    context_capture: ContextCapture | None = None,
    commercial_usage_producer: Any | None = None,
    agent_session_log: Any | None = None,
  ) -> None:
    if not isinstance(capability_execution, BoundCapabilityExecution):
      raise TypeError(
        "AgentSDKRunner requires a validated BoundCapabilityExecution"
      )
    capability_execution.validate()
    capability_bind = capability_execution.bind
    if capability_bind.capability_id != "session.driver":
      raise RuntimeError(
        "AgentSDKRunner requires a session.driver capability execution"
      )
    if capability_bind.adapter != "anthropic.agent_sdk":
      raise RuntimeError(
        "AgentSDKRunner requires the anthropic.agent_sdk adapter"
      )
    if capability_bind.provider != "anthropic":
      raise RuntimeError(
        "AgentSDKRunner requires an Anthropic capability execution"
      )

    self._log = event_log
    self._session_id = session_id or "no-session"
    self._sid = self._session_id[:12]
    self._session_started_at = float(started_at if started_at is not None else time.time())
    self._emit_session_recap = bool(emit_session_recap)
    self._sdk_config = sdk_config
    self._capability_execution = capability_execution
    self._secret_boundary = SecretBoundary.from_capability_execution(
      capability_execution
    )
    self._credential_env = _agent_sdk_credential_env(
      capability_execution
    )
    if max_tokens_override is not None:
      if (
        isinstance(max_tokens_override, bool)
        or not isinstance(max_tokens_override, int)
        or max_tokens_override <= 0
      ):
        raise ValueError("SDK max_tokens_override must be a positive integer")
      self._credential_env["CLAUDE_CODE_MAX_OUTPUT_TOKENS"] = str(
        max_tokens_override
      )
    self._max_tokens_override = max_tokens_override
    self._system_prompt = system_prompt
    self._disallowed_tools = list(disallowed_tools or sdk_config.disallowed_tools)
    mcp_config_metadata = mcp_server_configs
    self._run_identity_carrier = RunIdentityCarrier.from_optional(skill_run_id)
    self._mcp_run_identity_servers = frozenset(
      mcp_run_identity_servers or frozenset()
    )
    if self._mcp_run_identity_servers and self._run_identity_carrier is None:
      raise RunIdentityCarrierError(
        "run_identity_required",
        "run-bound MCP servers require a server-owned run identity",
      )
    if run_context is not None and self._run_identity_carrier is not None:
      context_carrier = RunIdentityCarrier.from_optional(run_context.run_id)
      if (
        context_carrier is None
        or context_carrier.run_id != self._run_identity_carrier.run_id
      ):
        raise RunIdentityCarrierError(
          "run_identity_mismatch",
          "SDK run context and transport carrier identities must match",
        )
    self._mcp_server_configs = inject_run_identity_into_mcp_server_configs(
      mcp_server_configs or {},
      server_names=self._mcp_run_identity_servers,
      carrier=self._run_identity_carrier,
    )
    self._deferred_mcp_server_configs = inject_run_identity_into_mcp_server_configs(
      getattr(
        mcp_config_metadata,
        "deferred_mcp_server_configs",
        getattr(sdk_config, "_deferred_mcp_server_configs", {}),
      ) or {},
      server_names=self._mcp_run_identity_servers,
      carrier=self._run_identity_carrier,
    )
    self._deferred_mcp_tool_ids = set(
      getattr(
        mcp_config_metadata,
        "deferred_mcp_tool_ids",
        getattr(sdk_config, "_deferred_mcp_tool_ids", set()),
      ) or set()
    )
    self._deferred_mcp_server_patterns = set(
      getattr(
        mcp_config_metadata,
        "deferred_mcp_server_patterns",
        getattr(sdk_config, "_deferred_mcp_server_patterns", set()),
      ) or set()
    )
    self._sdk_admission_enforced = bool(
      getattr(
        mcp_config_metadata,
        "sdk_admission_enforced",
        getattr(sdk_config, "_sdk_admission_enforced", False),
      )
    )
    self._sdk_advertised_mcp_tool_ids_by_server = {
      str(server_name): set(tool_ids)
      for server_name, tool_ids in (
        getattr(
          mcp_config_metadata,
          "advertised_mcp_tool_ids_by_server",
          getattr(sdk_config, "_advertised_mcp_tool_ids_by_server", {}),
        ) or {}
      ).items()
    }
    self._sdk_load_transaction_manager = getattr(
      mcp_config_metadata,
      "load_transaction_manager",
      getattr(sdk_config, "_load_transaction_manager", None),
    )
    self._pending_sdk_load: dict[str, set[str]] | None = None
    self._pending_sdk_load_transaction_ids: list[str] = []
    self._pending_sdk_load_transcript: list[str] = []
    self._sdk_rebuild_transcript: list[str] = []
    self._sdk_resume_session_id: str | None = None
    self._allowed_tools = list(allowed_tools or [])
    self._max_turns = max_turns
    self._on_usage = on_usage
    self._commercial_usage_producer = commercial_usage_producer
    self._commercial_usage_emitted = False
    self._sdk_provider_call_usage: dict[str, Any] | None = None
    self._pending_sdk_usage_deltas: list[dict[str, Any]] = []
    self._sdk_provider_usage_emitted_totals: dict[str, Any] = {
      "input_tokens": 0,
      "output_tokens": 0,
      "cache_read_input_tokens": 0,
      "cache_creation_input_tokens": 0,
      "provider_unit_deltas": {},
    }
    self._on_session_summary = on_session_summary
    self._on_late_usage_event = on_late_usage_event
    self._on_tool_result = on_tool_result
    self._on_tool_timing = on_tool_timing
    self._provider_id_for_tool = provider_id_for_tool
    self._on_tool_timing_accepts_user_id = _detect_user_id_param(on_tool_timing)
    self._on_tool_timing_accepts_context_surfaces = _detect_keyword_param(on_tool_timing, "context_surfaces")
    self._on_tool_timing_accepts_tool_call_id = _detect_keyword_param(on_tool_timing, "tool_call_id")
    self._on_tool_timing_accepts_request_id = _detect_keyword_param(on_tool_timing, "request_id")
    self._context_surfaces_provider = context_surfaces if callable(context_surfaces) else None
    self._context_surfaces_static = self._normalize_context_surfaces(None if callable(context_surfaces) else context_surfaces)
    self._context_capture = context_capture
    self._last_context_manifest_digest: str | None = None
    self._context_manifest_invocation = 0
    self._pending_tool_calls: Dict[str, _sdk_runner_stream.ToolCallInfo] = {}
    self._active_tool_use: _sdk_runner_stream._ActiveToolUse | None = None
    self._suppress_text_after_accepted_ui_blocks = False
    self._active_skill_allow: set[str] = set()
    self._active_skill_deny: set[str] = set()
    self._active_skill_report_doors: dict[str, str] = {}
    self._query_iter: Any = None
    self._usage: Dict[str, Any] = {
      "capability_bind": capability_bind.receipt(),
      "provider_reported_model": None,
      "input_tokens": 0,
      "output_tokens": 0,
      "reasoning_tokens_observed": 0,
      "cache_creation_input_tokens": 0,
      "cache_read_input_tokens": 0,
      "provider_reported_cost_usd": None,
      "provider_unit_deltas": {},
    }
    self._num_turns = 0
    self._stream_terminal_emitted = False
    self._effective_model = capability_bind.upstream_model
    self._request_id = str(sdk_config.request_id or uuid.uuid4())
    (
      self._usage_user_id,
      self._rate_table_version,
      self._billing_mode,
      self._channel,
    ) = normalize_identity(
      sdk_config.user_id,
      sdk_config.rate_table_version,
      sdk_config.billing_mode,
      sdk_config.channel,
    )
    self._parent_aggregator = _parent_aggregator
    self._aggregator = _parent_aggregator or _UsageAggregator(
      user_id=self._usage_user_id,
      session_id=self._session_id,
      request_id=self._request_id,
      channel=self._channel,
      rate_table_version=self._rate_table_version,
      billing_mode=self._billing_mode,
    )
    self._summary_emitted = False
    self._session = session
    self._agent_session_log = agent_session_log
    self._selected_content_bindings: tuple[SelectedContentBinding, ...] = ()
    self._selected_content_bindings_bound = False
    self._approval_store = store or getattr(session, "approval_store", None)
    self._approval_policy = policy or getattr(session, "approval_policy", None)
    self._run_context = run_context
    self._skill_run_id = skill_run_id
    self._workspace_dir = workspace_dir
    self._batch_id = str(batch_id).strip() if batch_id is not None and str(batch_id).strip() else None

  @property
  def capability_execution(self) -> BoundCapabilityExecution:
    return self._capability_execution

  def bind_selected_content(
    self,
    bindings: tuple[SelectedContentBinding, ...],
  ) -> None:
    if self._selected_content_bindings_bound:
      raise RuntimeError("selected content is already bound for this runner")
    if not isinstance(bindings, tuple) or not all(
      isinstance(binding, SelectedContentBinding) for binding in bindings
    ):
      raise TypeError("selected content must be an immutable binding tuple")
    self._selected_content_bindings = bindings
    self._selected_content_bindings_bound = True

  @staticmethod
  def _normalize_context_surfaces(surfaces: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    return _sdk_runner_context.normalize_context_surfaces(surfaces)

  def _context_surface_records(self) -> list[dict[str, Any]]:
    return _sdk_runner_context.context_surface_records(self, logger=log)

  def _append(self, event: Dict[str, Any]) -> None:
    payload = sanitize_tool_event(
      event,
      sink="event_log",
      boundary=self._secret_boundary,
    )
    pid = gateway_product_id()
    if pid is not None:
      payload["product_id"] = pid
    if payload.get("type") in {"stream_complete", "error"}:
      emit_recap_then_terminal(
        self._log,
        payload,
        session_id=self._session_id,
        started_at=self._session_started_at,
        emit_recap=self._emit_session_recap,
      )
      return
    self._log.append(payload)

  async def _call_on_usage(
    self,
    usage_event: UsageEvent,
    *,
    usage_state: str = "succeeded",
    emit_commercial: bool = True,
  ) -> None:
    await _sdk_runner_context.call_on_usage(
      self,
      usage_event,
      logger=log,
      usage_state=usage_state,
      emit_commercial=emit_commercial,
    )

  async def _call_on_late_usage_event(self, usage_event: UsageEvent) -> None:
    await _sdk_runner_context.call_on_late_usage_event(self, usage_event, logger=log)

  async def _call_on_session_summary(self, summary: SessionUsageSummary) -> None:
    await _sdk_runner_context.call_on_session_summary(self, summary, logger=log)

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
    _sdk_runner_context.call_on_tool_timing(
      self,
      tool_name=tool_name,
      server=server,
      duration_ms=duration_ms,
      is_error=is_error,
      result_bytes=result_bytes,
      tool_call_id=tool_call_id,
      request_id=request_id,
      logger=log,
    )

  async def _call_on_tool_result(self, ctx: ToolResultContext) -> List[Dict[str, Any]]:
    return await _sdk_runner_context.call_on_tool_result(self, ctx, logger=log)

  def _build_prompt(self, messages: List[Dict[str, Any]]) -> str:
    return _sdk_runner_context.build_prompt(messages)

  def _make_result_entry(
    self,
    tool_call_id: str,
    result: Any | None,
    error: Dict[str, Any] | None,
  ) -> Dict[str, Any]:
    return _sdk_runner_context.make_result_entry(tool_call_id, result, error)

  def _format_additional_context(
    self,
    *,
    tool_name: str,
    result_entry: Dict[str, Any],
    extra_blocks: Sequence[Dict[str, Any]],
  ) -> str | None:
    return _sdk_runner_context.format_additional_context(
      tool_name=tool_name,
      result_entry=result_entry,
      extra_blocks=extra_blocks,
    )

  async def _build_hook_additional_context(
    self,
    *,
    tool_call_id: str,
    tool_name: str,
    tool_input: Dict[str, Any],
    result: Any | None,
    error: Dict[str, Any] | None,
  ) -> str | None:
    return await _sdk_runner_context.build_hook_additional_context(
      self,
      tool_call_id=tool_call_id,
      tool_name=tool_name,
      tool_input=tool_input,
      result=result,
      error=error,
      logger=log,
    )

  def _effective_disallowed_tools(self) -> set[str]:
    return _SDKToolAdmissionSet(
      list(
        (set(self._disallowed_tools) - self._active_skill_allow)
        | self._active_skill_deny
      ),
      self,
    )

  @staticmethod
  def _sdk_tool_entry_names(entries: Any) -> set[str]:
    if not isinstance(entries, (list, tuple, set, frozenset)):
      return set()
    names: set[str] = set()
    for entry in entries:
      if isinstance(entry, str):
        name = entry.strip()
      elif isinstance(entry, dict):
        name = str(entry.get("name") or "").strip()
      else:
        name = ""
      if name:
        names.add(name)
    return names

  def _capture_sdk_load_signal(self, result: dict[str, Any]) -> None:
    raw_servers = result.pop("_load_servers", None)
    raw_transaction_id = result.pop("_load_transaction_id", None)
    transaction_id = str(raw_transaction_id or "").strip()
    if not isinstance(raw_servers, list):
      if transaction_id and self._sdk_load_transaction_manager is not None:
        self._sdk_load_transaction_manager.discard(transaction_id)
      return
    servers = {
      str(server_name).strip()
      for server_name in raw_servers
      if str(server_name or "").strip()
    }
    if not servers:
      if transaction_id and self._sdk_load_transaction_manager is not None:
        self._sdk_load_transaction_manager.discard(transaction_id)
      return
    tools_by_server: dict[str, set[str]] = {server_name: set() for server_name in servers}
    raw_new_tools = result.get("new_tools")
    if isinstance(raw_new_tools, dict):
      for server_name, entries in raw_new_tools.items():
        normalized_server = str(server_name or "").strip()
        if normalized_server in tools_by_server:
          tools_by_server[normalized_server].update(
            self._sdk_tool_entry_names(entries)
          )
    loaded_tools = self._sdk_tool_entry_names(result.get("loaded_tools"))
    if loaded_tools:
      for server_name in servers:
        tools_by_server[server_name].update(loaded_tools)
    public_result = {
      key: value
      for key, value in result.items()
      if not str(key).startswith("_")
    }
    self._pending_sdk_load_transcript.append(
      "TOOL load_tools RESULT: "
      + json.dumps(public_result, sort_keys=True, default=str)
    )
    if transaction_id:
      self._pending_sdk_load_transaction_ids.append(transaction_id)
    if self._pending_sdk_load is None:
      self._pending_sdk_load = tools_by_server
      return
    for server_name, tool_names in tools_by_server.items():
      self._pending_sdk_load.setdefault(server_name, set()).update(tool_names)

  def _discard_pending_sdk_load(self) -> None:
    manager = self._sdk_load_transaction_manager
    if manager is not None:
      for transaction_id in self._pending_sdk_load_transaction_ids:
        manager.discard(transaction_id)
    self._pending_sdk_load = None
    self._pending_sdk_load_transaction_ids.clear()
    self._pending_sdk_load_transcript.clear()

  def _pending_sdk_load_error(
    self,
    pending: dict[str, set[str]] | None,
  ) -> dict[str, Any] | None:
    if not pending:
      return {
        "code": "sdk_tool_rebuild_exhausted",
        "message": "load_tools did not admit any deferred SDK tool.",
      }
    changes_admission = False
    for server_name, tool_names in pending.items():
      if server_name not in self._mcp_server_configs:
        if server_name not in self._deferred_mcp_server_configs:
          return {
            "code": "sdk_tool_rebuild_unavailable",
            "message": (
              "No deferred SDK configuration is available for MCP server "
              f"'{server_name}'."
            ),
          }
        if not tool_names:
          return {
            "code": "sdk_tool_rebuild_unavailable",
            "message": (
              "No advertised tool schema inventory is available for deferred "
              f"MCP server '{server_name}'."
            ),
          }
        changes_admission = True
      candidate_ids = {
        (
          tool_name
          if tool_name.startswith(f"mcp__{server_name}__")
          else f"mcp__{server_name}__{tool_name}"
        )
        for tool_name in tool_names
      }
      if candidate_ids & self._deferred_mcp_tool_ids:
        changes_admission = True
    if not changes_admission:
      return {
        "code": "sdk_tool_rebuild_exhausted",
        "message": "load_tools did not admit any deferred SDK tool.",
      }
    return None

  def _apply_pending_sdk_load(self) -> tuple[bool, str | None]:
    pending = self._pending_sdk_load
    pending_error = self._pending_sdk_load_error(pending)
    if pending_error is not None:
      self._discard_pending_sdk_load()
      return False, (
        f"{pending_error['code']}: {pending_error['message']}"
      )
    assert pending is not None

    manager = self._sdk_load_transaction_manager
    if self._pending_sdk_load_transaction_ids and manager is None:
      self._discard_pending_sdk_load()
      return False, (
        "sdk_load_transaction_unavailable: staged load state cannot be committed"
      )
    if manager is not None:
      unavailable = [
        transaction_id
        for transaction_id in self._pending_sdk_load_transaction_ids
        if not manager.contains(transaction_id)
      ]
      if unavailable:
        self._discard_pending_sdk_load()
        return False, (
          "sdk_load_transaction_unavailable: staged load state is no longer available"
        )

    new_configs: dict[str, Any] = {}
    admitted_ids: set[str] = set()
    admitted_by_server: dict[str, set[str]] = {}
    for server_name, tool_names in pending.items():
      if server_name not in self._mcp_server_configs:
        config = self._deferred_mcp_server_configs.get(server_name)
        assert config is not None
        new_configs[server_name] = config
      server_admitted_ids = {
        (
          tool_name
          if tool_name.startswith(f"mcp__{server_name}__")
          else f"mcp__{server_name}__{tool_name}"
        )
        for tool_name in tool_names
      }
      admitted_ids.update(server_admitted_ids)
      admitted_by_server[server_name] = server_admitted_ids

    removable_ids = admitted_ids & self._deferred_mcp_tool_ids
    server_patterns = {
      f"mcp__{server_name}__*"
      for server_name in new_configs
    } & self._deferred_mcp_server_patterns
    if manager is not None:
      try:
        for transaction_id in self._pending_sdk_load_transaction_ids:
          manager.commit(transaction_id)
      except Exception as exc:
        self._discard_pending_sdk_load()
        return False, f"sdk_load_transaction_failed: {exc}"

    self._mcp_server_configs.update(new_configs)
    for server_name in new_configs:
      self._deferred_mcp_server_configs.pop(server_name, None)
      self._sdk_advertised_mcp_tool_ids_by_server[server_name] = set(
        admitted_by_server.get(server_name, set())
      )
    for server_name, server_admitted_ids in admitted_by_server.items():
      if server_name in self._sdk_advertised_mcp_tool_ids_by_server:
        self._sdk_advertised_mcp_tool_ids_by_server[server_name].update(
          server_admitted_ids
        )
    self._deferred_mcp_tool_ids.difference_update(removable_ids)
    self._deferred_mcp_server_patterns.difference_update(server_patterns)
    removed = removable_ids | server_patterns
    self._disallowed_tools = [
      tool_name for tool_name in self._disallowed_tools if tool_name not in removed
    ]
    self._sdk_rebuild_transcript.extend(self._pending_sdk_load_transcript)
    self._pending_sdk_load = None
    self._pending_sdk_load_transaction_ids.clear()
    self._pending_sdk_load_transcript.clear()
    return True, None

  def _activate_skill_deny(self, tool_names: Any) -> None:
    if isinstance(tool_names, str):
      candidates = [tool_names]
    elif isinstance(tool_names, (list, tuple, set, frozenset)):
      candidates = list(tool_names)
    else:
      return
    denied = {normalized for name in candidates if (normalized := str(name or "").strip())}
    self._active_skill_deny = denied

  def _activate_skill_allow(self, tool_names: Any) -> None:
    if isinstance(tool_names, str):
      candidates = [tool_names]
    elif isinstance(tool_names, (list, tuple, set, frozenset)):
      candidates = list(tool_names)
    else:
      return
    self._active_skill_allow = {
      normalized
      for name in candidates
      if (normalized := str(name or "").strip())
    }

  def _activate_skill_report_doors(self, value: Any) -> None:
    if isinstance(value, dict):
      self._active_skill_report_doors = {
        str(tool_name).strip(): str(skill_name).strip()
        for tool_name, skill_name in value.items()
        if str(tool_name).strip() and str(skill_name).strip()
      }
      return
    if value is not None:
      self._active_skill_report_doors = {}

  def _clear_active_skill_if_report_door_completed(
    self,
    *,
    tool_name: str,
    result: Any,
    error: Dict[str, Any] | None,
  ) -> bool:
    if error is not None:
      return False
    normalized_tool_name = str(tool_name or "").strip()
    expected_skill = self._active_skill_report_doors.get(normalized_tool_name)
    if not _is_report_door_clear_event(
      {
        "type": "tool_call_complete",
        "tool_name": normalized_tool_name,
        "result": result,
        "error": None,
      },
      expected_skill=expected_skill,
      success_statuses=_REPORT_DOOR_CLEAR_SUCCESS_STATUSES,
    ):
      return False
    if current_skill() != expected_skill:
      return False
    clear_current_skill()
    self._active_skill_allow.clear()
    self._active_skill_deny.clear()
    self._active_skill_report_doors.clear()
    return True

  def _consume_private_tool_result_fields(self, result: Any, *, tool_name: str | None = None) -> Any:
    if isinstance(result, dict):
      if tool_name == _TRUSTED_SDK_LOAD_TOOLS_ID:
        self._capture_sdk_load_signal(result)
      else:
        result.pop("_load_servers", None)
        result.pop("_load_transaction_id", None)
      # This runtime citation projection is private, so strip it before the
      # result reaches the model or the durable transcript.
      result.pop(_WORKFLOW_EVIDENCE_PROJECTION_RESULT_KEY, None)
      self._activate_skill_report_doors(result.pop(_ACTIVE_SKILL_REPORT_DOORS_RESULT_KEY, None))
      self._activate_skill_allow(result.pop(_ACTIVE_SKILL_ALLOW_RESULT_KEY, None))
      self._activate_skill_deny(result.pop(_ACTIVE_SKILL_DENY_RESULT_KEY, None))
      if tool_name is not None:
        self._clear_active_skill_if_report_door_completed(
          tool_name=tool_name,
          result=result,
          error=None,
        )
    return result

  async def _post_tool_use_hook(
    self,
    input_data: Dict[str, Any],
    tool_use_id: str | None,
    _context: Any,
  ) -> Dict[str, Any]:
    tool_name = str(input_data.get("tool_name") or input_data.get("name") or "")
    hook_session_id = str(input_data.get("session_id") or "").strip()
    if hook_session_id:
      self._sdk_resume_session_id = hook_session_id
    tool_input = _as_dict(input_data.get("tool_input") or input_data.get("input"))
    raw_result = input_data.get(
      "tool_response",
      input_data.get(
        "result",
        input_data.get("tool_result", input_data.get("output")),
      ),
    )
    result = _parse_result_payload(raw_result)
    if isinstance(result, dict) and isinstance(result.get("content"), list):
      nested_result = _parse_result_payload(_extract_text(result["content"]))
      if nested_result not in (None, ""):
        result = nested_result
    elif isinstance(result, list):
      nested_result = _parse_result_payload(_extract_text(result))
      if nested_result not in (None, ""):
        result = nested_result
    had_private_load_signal = (
      tool_name == _TRUSTED_SDK_LOAD_TOOLS_ID
      and isinstance(result, dict)
      and "_load_servers" in result
    )
    result = self._consume_private_tool_result_fields(result, tool_name=tool_name)
    rebuild_error: dict[str, Any] | None = None
    if had_private_load_signal and not self._sdk_resume_session_id:
      self._discard_pending_sdk_load()
      rebuild_error = {
        "code": "sdk_tool_rebuild_transcript_unavailable",
        "message": (
          "The SDK session identity required to preserve the transcript was "
          "not available; no deferred tool was advertised."
        ),
      }
      result = {"error": rebuild_error}
    elif had_private_load_signal:
      rebuild_error = self._pending_sdk_load_error(self._pending_sdk_load)
      if rebuild_error is not None:
        self._discard_pending_sdk_load()
        result = {"error": rebuild_error}
    additional_context = await self._build_hook_additional_context(
      tool_call_id=str(tool_use_id or input_data.get("tool_use_id") or ""),
      tool_name=tool_name,
      tool_input=tool_input,
      result=result,
      error=None,
    )
    safe_result = sanitize_boundary_value(
      result,
      sink="sdk_model_tool_result",
      boundary=self._secret_boundary,
    )
    safe_additional_context = (
      sanitize_boundary_value(
        additional_context,
        sink="sdk_model_tool_result",
        boundary=self._secret_boundary,
      )
      if additional_context
      else None
    )
    if (
      safe_result == result
      and not safe_additional_context
      and not had_private_load_signal
    ):
      return {}
    response: dict[str, Any] = {}
    hook_output: dict[str, Any] = {
      "hookEventName": "PostToolUse",
    }
    if safe_result != result or had_private_load_signal:
      hook_output["updatedMCPToolOutput"] = safe_result
    if safe_additional_context:
      hook_output["additionalContext"] = safe_additional_context
    if had_private_load_signal:
      if rebuild_error is None:
        response["continue_"] = False
        response["stopReason"] = (
          "load_tools changed the advertised MCP tool set; the gateway will resume "
          "this SDK session with rebuilt options"
        )
    response["hookSpecificOutput"] = hook_output
    return response

  async def _post_tool_use_failure_hook(
    self,
    input_data: Dict[str, Any],
    tool_use_id: str | None,
    _context: Any,
  ) -> Dict[str, Any]:
    tool_name = str(input_data.get("tool_name") or input_data.get("name") or "")
    tool_input = _as_dict(input_data.get("tool_input") or input_data.get("input"))
    error_message = _summarize_error_payload(
      input_data.get("error", input_data.get("result", input_data.get("message")))
    )
    error = {
      "code": str(input_data.get("code") or "tool_error"),
      "message": error_message,
    }
    additional_context = await self._build_hook_additional_context(
      tool_call_id=str(tool_use_id or input_data.get("tool_use_id") or ""),
      tool_name=tool_name,
      tool_input=tool_input,
      result=None,
      error=error,
    )
    safe_error = sanitize_boundary_value(
      error,
      sink="sdk_model_tool_error",
      boundary=self._secret_boundary,
    )
    safe_additional_context = (
      sanitize_boundary_value(
        additional_context,
        sink="sdk_model_tool_error",
        boundary=self._secret_boundary,
      )
      if additional_context
      else None
    )
    if not safe_additional_context and safe_error == error:
      return {}
    response: dict[str, Any] = {
      "hookSpecificOutput": {
        "hookEventName": "PostToolUseFailure",
      }
    }
    if safe_additional_context:
      response["hookSpecificOutput"]["additionalContext"] = safe_additional_context
    if safe_error != error:
      response["decision"] = "block"
      response["reason"] = str(safe_error)
    return response

  def _build_hooks(self, hook_matcher_cls: Any) -> Dict[str, List[Any]]:
    hooks: Dict[str, List[Any]] = {
      "PostToolUse": [hook_matcher_cls(hooks=[self._post_tool_use_hook])],
    }
    hooks["PostToolUseFailure"] = [hook_matcher_cls(hooks=[self._post_tool_use_failure_hook])]
    return hooks

  def _approval_lifecycle_configured(self) -> bool:
    return _sdk_runner_approval.approval_lifecycle_configured(
      store=self._approval_store,
      policy=self._approval_policy,
      session=self._session,
    )

  def _resolve_run_context(self) -> RunContext:
    return _sdk_runner_approval.resolve_run_context(
      run_context=self._run_context,
      run_id=(
        self._run_identity_carrier.run_id
        if self._run_identity_carrier is not None
        else None
      ),
      usage_user_id=self._usage_user_id,
      session=self._session,
      approval_policy=self._approval_policy,
      request_id=self._request_id,
      session_id=self._session_id,
      channel=self._channel,
    )

  def _resolve_tool_class(self, tool_name: str) -> str:
    return _sdk_runner_approval.resolve_tool_class(
      tool_name,
      policy_tool_name_fn=_policy_tool_name,
      server_for_tool_fn=_server_for_tool,
      resolve_server_policy_tool_class_fn=resolve_server_policy_tool_class,
    )

  def _redact_for_approval_request(self, tool_name: str, tool_input: Dict[str, Any]) -> tuple[dict[str, Any], str]:
    return _sdk_runner_approval.redact_for_approval_request(
      tool_name,
      tool_input,
      sha256_args_fn=sha256_args,
    )

  async def _await_user_approval_via_pending_tools(
    self,
    request: PolicyApprovalRequest,
    decision: PolicyApprovalDecision,
    *,
    nonce: str,
    batch_admission: Any | None = None,
  ) -> dict[str, Any] | None:
    return await _sdk_runner_approval.await_user_approval_via_pending_tools(
      session=self._session,
      approval_store=self._approval_store,
      request=request,
      decision=decision,
      nonce=nonce,
      append_event_fn=self._append,
      timeout_seconds=_approval_queue_timeout_seconds(decision.expiry_seconds),
      log=log,
      time_fn=time.time,
      batch_admission=batch_admission,
    )

  async def _can_use_tool_callback(self, tool_name: str, input_data: dict[str, Any], _context: Any) -> Any:
    if tool_name in MODEL_RUN_IDENTITY_LOCAL_TOOLS:
      import claude_agent_sdk

      deny_cls = getattr(claude_agent_sdk, "PermissionResultDeny")
      try:
        carrier = self._run_identity_carrier
        if carrier is None:
          raise RunIdentityCarrierError(
            "run_identity_required",
            f"Tool '{tool_name}' requires a server-owned run identity.",
          )
        context_run_id = validate_run_identity(
          self._resolve_run_context().run_id
        )
        if context_run_id != carrier.run_id:
          raise RunIdentityCarrierError(
            "run_identity_mismatch",
            f"Tool '{tool_name}' received conflicting run identities.",
          )
      except RunIdentityCarrierError as exc:
        return deny_cls(message=f"[{exc.code}] {exc}")
    return await _sdk_runner_approval.can_use_tool_callback(
      self,
      tool_name,
      input_data,
      _context,
      policy_owner_mismatch_fn=_policy_owner_mismatch,
      policy_tool_name_fn=_policy_tool_name,
      current_skill_fn=current_skill,
      replace_fn=replace,
      enrich_trade_approval_args_fn=enrich_trade_approval_args,
      build_approval_request_fn=build_approval_request,
      call_policy_safely_fn=call_policy_safely,
      apply_decision_to_request_fn=apply_decision_to_request,
      effective_trade_approval_expiry_seconds_fn=effective_trade_approval_expiry_seconds,
      approval_wait_seconds_fn=approval_settings.approval_wait_seconds,
      utc_now_fn=utc_now,
      uuid_hex_fn=lambda: uuid.uuid4().hex,
      os_urandom_fn=os.urandom,
    )

  async def _close_query_iterator(self) -> None:
    iterator = self._query_iter
    self._query_iter = None
    if iterator is None:
      return
    close_fn = getattr(iterator, "aclose", None)
    if close_fn is not None:
      try:
        await close_fn()
      except Exception:
        pass
      return
    close_fn = getattr(iterator, "close", None)
    if close_fn is None:
      return
    try:
      maybe_awaitable = close_fn()
      if asyncio.iscoroutine(maybe_awaitable):
        await maybe_awaitable
    except Exception:
      pass

  async def on_disconnect(self) -> None:
    try:
      await self._close_query_iterator()
    except Exception as exc:
      log.warning("[%s] query iterator close on disconnect failed (non-fatal): %s", self._sid, exc)

  def _update_usage(self, usage: Any, *, total_cost_usd: float | None = None, num_turns: int | None = None) -> None:
    usage_dict = _as_dict(usage)
    raw_input_tokens = int(usage_dict.get("input_tokens") or self._usage.get("input_tokens") or 0)
    self._usage["output_tokens"] = int(usage_dict.get("output_tokens") or self._usage.get("output_tokens") or 0)
    self._usage["reasoning_tokens_observed"] = int(
      usage_dict.get("reasoning_tokens")
      or usage_dict.get("reasoning_tokens_observed")
      or self._usage.get("reasoning_tokens_observed")
      or 0
    )
    self._usage["cache_creation_input_tokens"] = int(
      usage_dict.get("cache_creation_input_tokens") or self._usage.get("cache_creation_input_tokens") or 0
    )
    self._usage["cache_read_input_tokens"] = int(
      usage_dict.get("cache_read_input_tokens") or self._usage.get("cache_read_input_tokens") or 0
    )
    self._usage["input_tokens"] = raw_input_tokens
    self._usage["provider_unit_deltas"] = (
      dict(usage_dict.get("provider_unit_deltas") or {})
      or _server_tool_unit_deltas(usage_dict)
    )
    provider_reported_cost_usd = total_cost_usd
    if total_cost_usd is None:
      estimated = estimate_cost(
        self._effective_model,
        int(self._usage["input_tokens"]),
        int(self._usage["output_tokens"]),
        cache_read_tokens=int(self._usage["cache_read_input_tokens"]),
        cache_creation_tokens=int(self._usage["cache_creation_input_tokens"]),
      )
      total_cost_usd = estimated.total
    self._usage["estimated_cost"] = float(total_cost_usd or 0.0)
    self._usage["provider_reported_cost_usd"] = provider_reported_cost_usd
    if num_turns is not None:
      self._num_turns = int(num_turns)

  async def _emit_usage_hook(
    self, *, usage_state: str = "succeeded", emit_commercial: bool = True
  ) -> None:
    # The SDK exposes one cumulative ResultMessage per query, not provider-call
    # deltas. This event is therefore one SDK query operation; a terminal
    # path without a ResultMessage emits a zero observation for reconciliation.
    usage_event = UsageEvent(
      user_id=self._usage_user_id,
      session_id=self._session_id,
      request_id=self._request_id,
      parent_turn_id=None,
      timestamp=time.time(),
      model=self._effective_model,
      provider=self._capability_execution.bind.provider,
      capability_bind=self._capability_execution.bind.receipt(),
      provider_reported_model=(
        str(self._usage["provider_reported_model"])
        if self._usage.get("provider_reported_model") is not None else None
      ),
      input_tokens=int(self._usage.get("input_tokens") or 0),
      output_tokens=int(self._usage.get("output_tokens") or 0),
      reasoning_tokens_observed=int(self._usage.get("reasoning_tokens_observed") or 0),
      cache_read_tokens=int(self._usage.get("cache_read_input_tokens") or 0),
      cache_creation_tokens=int(self._usage.get("cache_creation_input_tokens") or 0),
      cost_usd=float(self._usage.get("estimated_cost") or 0.0),
      provider_reported_cost_usd=(
        str(self._usage["provider_reported_cost_usd"])
        if self._usage.get("provider_reported_cost_usd") is not None else None
      ),
      provider_unit_deltas=dict(self._usage.get("provider_unit_deltas") or {}) or None,
      rate_table_version=self._rate_table_version,
      billing_mode=self._billing_mode,
      channel=self._channel,
    )
    if emit_commercial and self._commercial_usage_producer is not None:
      await self._commercial_usage_producer.emit(
        usage_event, usage_state=usage_state
      )
      self._commercial_usage_emitted = True
    await self._call_on_usage(
      usage_event, usage_state=usage_state, emit_commercial=False
    )

  async def _emit_sdk_provider_call_usage(
    self, usage: dict[str, Any], *, usage_state: str = "succeeded"
  ) -> None:
    model = self._capability_execution.bind.upstream_model
    cost = estimate_cost(
      model,
      int(usage.get("input_tokens") or 0),
      int(usage.get("output_tokens") or 0),
      cache_read_tokens=int(usage.get("cache_read_input_tokens") or 0),
      cache_creation_tokens=int(usage.get("cache_creation_input_tokens") or 0),
    )
    event = UsageEvent(
      user_id=self._usage_user_id,
      session_id=self._session_id,
      request_id=self._request_id,
      parent_turn_id=None,
      timestamp=time.time(),
      model=model,
      provider=self._capability_execution.bind.provider,
      capability_bind=self._capability_execution.bind.receipt(),
      provider_reported_model=(
        str(usage["provider_reported_model"])
        if usage.get("provider_reported_model") is not None else None
      ),
      input_tokens=int(usage.get("input_tokens") or 0),
      output_tokens=int(usage.get("output_tokens") or 0),
      cache_read_tokens=int(usage.get("cache_read_input_tokens") or 0),
      cache_creation_tokens=int(usage.get("cache_creation_input_tokens") or 0),
      provider_unit_deltas=dict(usage.get("provider_unit_deltas") or {}) or None,
      cost_usd=float(cost.total),
      rate_table_version=self._rate_table_version,
      billing_mode=self._billing_mode,
      channel=self._channel,
    )
    producer = self._commercial_usage_producer
    if producer is None:
      return
    await producer.emit(event, usage_state=usage_state)
    for key in (
      "input_tokens",
      "output_tokens",
      "cache_read_input_tokens",
      "cache_creation_input_tokens",
    ):
      self._sdk_provider_usage_emitted_totals[key] = int(
        self._sdk_provider_usage_emitted_totals.get(key) or 0
      ) + int(usage.get(key) or 0)
    emitted_units = dict(
      self._sdk_provider_usage_emitted_totals.get("provider_unit_deltas") or {}
    )
    for unit_name, unit_count in dict(
      usage.get("provider_unit_deltas") or {}
    ).items():
      emitted_units[str(unit_name)] = int(
        emitted_units.get(str(unit_name)) or 0
      ) + int(unit_count or 0)
    self._sdk_provider_usage_emitted_totals["provider_unit_deltas"] = (
      emitted_units
    )
    self._commercial_usage_emitted = True

  def _pending_sdk_provider_usage_at_result(
    self,
    result_usage: dict[str, Any],
  ) -> dict[str, Any] | None:
    pending = self._sdk_provider_call_usage
    if pending is None:
      return None
    terminal_usage = dict(pending)
    emitted_totals = self._sdk_provider_usage_emitted_totals
    for key in (
      "input_tokens",
      "output_tokens",
      "cache_read_input_tokens",
      "cache_creation_input_tokens",
    ):
      if key not in result_usage:
        continue
      cumulative_value = int(result_usage.get(key) or 0)
      emitted_value = int(emitted_totals.get(key) or 0)
      if cumulative_value < emitted_value:
        raise RuntimeError(
          "agent_sdk_result_usage_regressed: "
          f"{key}={cumulative_value} is below emitted={emitted_value}"
        )
      terminal_usage[key] = cumulative_value - emitted_value

    result_units = _server_tool_unit_deltas(result_usage)
    if result_units:
      emitted_units = dict(emitted_totals.get("provider_unit_deltas") or {})
      terminal_units: dict[str, int] = {}
      for unit_name, cumulative_count in result_units.items():
        emitted_count = int(emitted_units.get(unit_name) or 0)
        normalized_count = int(cumulative_count or 0)
        if normalized_count < emitted_count:
          raise RuntimeError(
            "agent_sdk_result_usage_regressed: "
            f"provider_unit_deltas.{unit_name}={normalized_count} "
            f"is below emitted={emitted_count}"
          )
        residual_count = normalized_count - emitted_count
        if residual_count:
          terminal_units[unit_name] = residual_count
      terminal_usage["provider_unit_deltas"] = terminal_units
    return terminal_usage

  async def run(
    self,
    messages: list[dict],
    system_prompt: str | None = None,
    max_turns: int | None = None,
  ) -> None:
    if self._summary_emitted:
      raise RuntimeError("AgentSDKRunner is single-use; construct a new runner for subsequent runs")
    if self._selected_content_bindings:
      append = getattr(self._agent_session_log, "append", None)
      user_content = next(
        (
          message.get("content")
          for message in reversed(messages)
          if message.get("role") == "user"
        ),
        None,
      )
      if not callable(append) or user_content is None:
        raise RuntimeError(
          "Selected content was not durably committed before model work."
        )
      await append(build_user_message_event(
        content=user_content,
        client_kind=self._channel or "chat",
        received_at=time.time(),
        selected_content=serialize_selected_content_bindings(
          self._selected_content_bindings
        ),
      ))
    self._capability_execution.validate()
    capability_bind = self._capability_execution.bind
    effective_model = capability_bind.upstream_model
    self._effective_model = effective_model
    if self._commercial_usage_producer is not None:
      commercial_guard = getattr(self._commercial_usage_producer, "assert_work_allowed", None)
      if callable(commercial_guard):
        commercial_guard(self._billing_mode)
    _validate_sdk_version()
    try:
      import claude_agent_sdk
    except ImportError as exc:
      self._append({"type": "error", "error": "claude-agent-sdk dependency is required"})
      raise RuntimeError("claude-agent-sdk dependency is required") from exc

    prompt_text = self._build_prompt(messages)
    effective_system_prompt = _join_system_prompt(system_prompt or self._system_prompt)
    self._context_manifest_invocation += 1
    if self._context_capture is not None:
      surfaces = self._context_surface_records()
      try:
        system_prompt_hash = await asyncio.to_thread(
          self._context_capture.persist,
          surfaces=surfaces,
          rendered_system_prompt=effective_system_prompt or None,
        )
        digest = canonical_manifest_digest(surfaces, system_prompt_hash)
        if digest != self._last_context_manifest_digest:
          self._append(build_context_manifest_event(
            surfaces=surfaces,
            system_prompt_hash=system_prompt_hash,
            session_id=self._session_id,
            request_id=self._request_id,
            turn=None,
            invocation=self._context_manifest_invocation,
          ))
          self._last_context_manifest_digest = digest
      except Exception as exc:
        log.warning(
          "[%s] context capture failed; manifest suppressed | exception_type=%s",
          self._sid,
          type(exc).__name__,
        )
    effort = capability_bind.effort
    effort_options: Dict[str, Any] = {}
    if effort == "none":
      effort_options["thinking"] = {"type": "disabled"}
    elif effort in {"low", "medium", "high", "max"}:
      effort_options["effort"] = effort
    else:
      raise RuntimeError(f"unsupported agent-sdk effort: {effort}")

    hooks = self._build_hooks(getattr(claude_agent_sdk, "HookMatcher"))
    options_kwargs: Dict[str, Any] = {
      "system_prompt": effective_system_prompt or None,
      "mcp_servers": dict(self._mcp_server_configs),
      "allowed_tools": list(self._allowed_tools),
      "continue_conversation": False,
      "max_turns": max_turns if max_turns is not None else self._max_turns,
      "max_budget_usd": self._sdk_config.max_budget_usd,
      "disallowed_tools": list(self._disallowed_tools),
      "model": effective_model or None,
      "cwd": str(self._sdk_config.cwd) if self._sdk_config.cwd is not None else None,
      "include_partial_messages": True,
      "hooks": hooks or None,
      "can_use_tool": self._can_use_tool_callback,
      "env": dict(self._credential_env),
      **effort_options,
    }
    options_kwargs = {key: value for key, value in options_kwargs.items() if value is not None}

    try:
      result_seen = False
      result_error: RuntimeError | None = None
      query_iter = _SDKSegmentQueryIterator(
        runner=self,
        sdk_module=claude_agent_sdk,
        initial_prompt=prompt_text,
        options_kwargs=options_kwargs,
      )
      self._query_iter = query_iter
      async for message in query_iter:
        if hasattr(message, "event"):
          self._handle_stream_event(_as_dict(getattr(message, "event")))
          while self._pending_sdk_usage_deltas:
            await self._emit_sdk_provider_call_usage(
              self._pending_sdk_usage_deltas.pop(0)
            )
            commercial_guard = getattr(
              self._commercial_usage_producer, "assert_work_allowed", None
            )
            if callable(commercial_guard):
              # The SDK iterator cannot advance to another provider turn after
              # a durability incident observed at this response boundary.
              commercial_guard(self._billing_mode)
          continue

        if hasattr(message, "duration_ms") and hasattr(message, "num_turns"):
          result_seen = True
          result_subtype = str(_get_attr(message, "subtype") or "")
          result_is_error = bool(_get_attr(message, "is_error"))
          result_succeeded = (
            result_subtype == "success"
            and not result_is_error
          )
          result_usage = _as_dict(_get_attr(message, "usage"))
          result_usage_state = (
            "succeeded" if result_succeeded else "failed_billable"
          )
          commercial_usage_previously_emitted = (
            self._commercial_usage_emitted
          )
          pending_provider_usage = self._pending_sdk_provider_usage_at_result(
            result_usage
          )
          if (
            pending_provider_usage is not None
            and commercial_usage_previously_emitted
          ):
            # Earlier completed provider calls were already emitted at their
            # message_delta boundaries. Emit only the residual final call so
            # the cumulative ResultMessage cannot duplicate those calls.
            await self._emit_sdk_provider_call_usage(
              pending_provider_usage,
              usage_state=result_usage_state,
            )
            self._sdk_provider_call_usage = None
          self._update_usage(
            result_usage,
            total_cost_usd=_get_attr(message, "total_cost_usd"),
            num_turns=_get_attr(message, "num_turns"),
          )
          await self._emit_usage_hook(
            usage_state=result_usage_state,
            emit_commercial=not commercial_usage_previously_emitted,
          )
          if (
            pending_provider_usage is not None
            and not commercial_usage_previously_emitted
          ):
            # With no prior per-call emission, the cumulative ResultMessage is
            # the single commercial observation for the whole query.
            self._sdk_provider_call_usage = None
          if result_succeeded:
            self._flush_pending_tool_calls(outcome="success")
            self._emit_stream_complete()
          elif result_subtype == "error_max_turns":
            self._flush_pending_tool_calls(outcome="cancelled")
            configured_max_turns = options_kwargs.get("max_turns")
            if isinstance(configured_max_turns, int):
              self._append({
                "type": "max_turns_reached",
                "turn_count": int(_get_attr(message, "num_turns") or 0),
                "max_turns": configured_max_turns,
              })
            self._emit_stream_complete(
              terminal_disposition="interrupted",
              reason="max_turns_reached",
            )
          elif result_subtype == "error_max_budget_usd":
            self._flush_pending_tool_calls(outcome="cancelled")
            configured_max_budget = self._sdk_config.max_budget_usd
            if isinstance(configured_max_budget, (int, float)):
              self._append({
                "type": "budget_exceeded",
                "total_cost": float(_get_attr(message, "total_cost_usd") or 0.0),
                "budget": float(configured_max_budget),
                "reason": "sdk_max_budget_usd",
              })
            self._emit_stream_complete(
              terminal_disposition="interrupted",
              reason="budget_exceeded",
            )
          else:
            result_detail = str(
              _get_attr(message, "result")
              or result_subtype
              or "unknown SDK result failure"
            )
            result_error = RuntimeError(
              f"agent_sdk_result_error: {result_detail}"
            )
            self._flush_pending_tool_calls(outcome="tool_error")
            self._append({"type": "error", "error": str(result_error)})
            self._stream_terminal_emitted = True
          break

        if hasattr(message, "subtype") and hasattr(message, "data"):
          self._handle_system_message(message)
          continue

        if hasattr(message, "model") and hasattr(message, "content"):
          self._handle_assistant_message(message)
          continue

        if hasattr(message, "content"):
          self._handle_user_message(message)

      if result_error is not None:
        raise result_error
      if not result_seen:
        raise RuntimeError(
          "agent_sdk_stream_ended_without_result: SDK iterator ended before "
          "a ResultMessage proved completion"
        )
    except CommercialUsageCircuitOpen as exc:
      await self._close_query_iterator()
      self._flush_pending_tool_calls(outcome="usage_durability_blocked")
      self._append({"type": "usage_durability_blocked", "error": str(exc)})
      self._append({
        "type": "error",
        "error": f"usage_durability_blocked: {exc}",
      })
      self._stream_terminal_emitted = True
      raise
    except asyncio.CancelledError:
      usage_cleanup_error: Exception | None = None
      try:
        if self._commercial_usage_producer is not None and self._sdk_provider_call_usage is not None:
          await self._emit_sdk_provider_call_usage(
            self._sdk_provider_call_usage, usage_state="canceled"
          )
          self._sdk_provider_call_usage = None
        elif self._commercial_usage_producer is not None and not self._commercial_usage_emitted:
          await self._emit_usage_hook(usage_state="canceled")
      except Exception as exc:
        usage_cleanup_error = exc
      await self._close_query_iterator()
      if usage_cleanup_error is None:
        self._flush_pending_tool_calls(outcome="cancelled")
        self._emit_stream_complete(
          terminal_disposition="interrupted",
          reason="cancelled",
        )
      else:
        interruption_reason = (
          "usage_durability_blocked"
          if isinstance(usage_cleanup_error, CommercialUsageCircuitOpen)
          else "tool_error"
        )
        self._flush_pending_tool_calls(outcome=interruption_reason)
        if isinstance(usage_cleanup_error, CommercialUsageCircuitOpen):
          self._append({
            "type": "usage_durability_blocked",
            "error": str(usage_cleanup_error),
          })
        self._append({
          "type": "error",
          "error": (
            "cancellation_usage_cleanup_failed: "
            f"{type(usage_cleanup_error).__name__}: "
            f"{usage_cleanup_error}"
          ),
        })
        self._stream_terminal_emitted = True
      raise
    except Exception as exc:
      usage_cleanup_error = None
      try:
        if self._commercial_usage_producer is not None and self._sdk_provider_call_usage is not None:
          await self._emit_sdk_provider_call_usage(
            self._sdk_provider_call_usage, usage_state="failed_billable"
          )
          self._sdk_provider_call_usage = None
        elif self._commercial_usage_producer is not None and not self._commercial_usage_emitted:
          await self._emit_usage_hook(usage_state="failed_unbilled")
      except Exception as cleanup_exc:
        usage_cleanup_error = cleanup_exc
      await self._close_query_iterator()
      if not self._stream_terminal_emitted:
        interruption_reason = (
          "usage_durability_blocked"
          if isinstance(usage_cleanup_error, CommercialUsageCircuitOpen)
          else "tool_error"
        )
        self._flush_pending_tool_calls(outcome=interruption_reason)
        if isinstance(usage_cleanup_error, CommercialUsageCircuitOpen):
          self._append({
            "type": "usage_durability_blocked",
            "error": str(usage_cleanup_error),
          })
        error_message = str(exc)
        if usage_cleanup_error is not None:
          error_message = (
            f"{error_message}; usage_cleanup_failed: "
            f"{type(usage_cleanup_error).__name__}: "
            f"{usage_cleanup_error}"
          )
        self._append({"type": "error", "error": error_message})
        self._stream_terminal_emitted = True
      raise
    finally:
      clear_current_skill()
      self._active_skill_allow.clear()
      self._active_skill_deny.clear()
      self._active_skill_report_doors.clear()
      if self._parent_aggregator is None:
        try:
          await self._aggregator.close()
          summary = await self._aggregator.snapshot(
            ended_at=time.time(),
            drain_complete=True,
            in_flight_task_count=0,
            context_surfaces=self._context_surface_records(),
          )
          self._summary_emitted = True
          await self._call_on_session_summary(summary)
        except Exception as exc:
          log.warning(
            "[%s] session summary emission failed (non-fatal) | exception_type=%s",
            self._sid,
            type(exc).__name__,
          )
      else:
        self._summary_emitted = True
      await self._close_query_iterator()


__all__ = ["AgentSDKRunner"]
