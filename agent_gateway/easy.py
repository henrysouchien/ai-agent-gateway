from __future__ import annotations

import asyncio
import inspect
import logging
import os
import secrets
from pathlib import Path
from typing import Any, Awaitable, Callable

from fastapi import FastAPI

from ._provider_utils import _get_default_model_for_provider, _resolve_provider
from .auth import CredentialsRefreshResolver, CredentialsResolver
from .code_execution import CodeExecutionConfig, build_code_execution
from .commercial_work_start import CommercialWorkStartGate
from .mcp_client import McpClientManager
from .multi_user.billing import DEFAULT_USAGE_DLQ_PATH, SessionUsageSummary, UsageEvent, UsageLedger
from .providers import AnthropicProvider, ModelProvider
from .rates import load_rate_table
from .runner import AgentRunner, ToolResultContext
from .server import ChatRequest, ChatRuntime, GatewayServerConfig, _make_request_approval, create_gateway_app
from .session import AuthManager, GatewaySession
from .skills import SkillLoader, SkillStateStore
from .task_registry import CoordinatorConfig
from .tool_dispatcher import LocalToolHandler, ToolDispatcher

log = logging.getLogger("agent_gateway.easy")


class _NullMcpClient:
  def is_mcp_tool(self, _name: str) -> bool:
    return False

  async def call_tool(self, name: str, _tool_input: dict[str, Any]):
    return None, {"code": "unknown_tool", "message": f"Unknown tool: {name}"}

  def get_tool_definitions(self) -> list[dict[str, Any]]:
    return []


def _mcp_config_path_from_env() -> Path | None:
  env_path = os.getenv("MCP_CONFIG_PATH", "").strip()
  return Path(env_path).expanduser() if env_path else None


def _call_commercial_usage_producer_factory(
  factory: Callable[..., Awaitable[Any] | Any],
  *,
  session: GatewaySession,
  request: ChatRequest,
  channel: str | None,
) -> Awaitable[Any] | Any:
  commercial_work_start = request.commercial_work_start
  try:
    signature = inspect.signature(factory)
  except (TypeError, ValueError):
    return factory(session, request, channel)
  params = signature.parameters
  advertised = params.get("commercial_work_start")
  legacy_args = (session, request, channel)
  if advertised is not None and advertised.kind == advertised.POSITIONAL_ONLY:
    args = (*legacy_args, commercial_work_start)
    signature.bind(*args)
    return factory(*args)
  if advertised is not None or any(
    param.kind == param.VAR_KEYWORD for param in params.values()
  ):
    call_kwargs = {"commercial_work_start": commercial_work_start}
    signature.bind(*legacy_args, **call_kwargs)
    return factory(*legacy_args, **call_kwargs)
  signature.bind(*legacy_args)
  return factory(*legacy_args)


def create_agent(
  system_prompt: str | list[tuple[str, bool]],
  *,
  provider: str | ModelProvider = "anthropic",
  model: str | None = None,
  provider_config: dict[str, Any] | None = None,
  api_key: str | None = None,
  auth_token: str | None = None,
  rates_file: Path | None = None,
  max_tokens: int = 16_000,
  mcp_servers: dict[str, dict[str, Any]] | None = None,
  mcp_config_path: str | Path | None = None,
  mcp_timeout_overrides: dict[str, int] | None = None,
  mcp_default_tool_timeout: int = 30,
  mcp_session_inject_servers: set[str] | None = None,
  mcp_strip_input_fields: set[str] | None = None,
  tool_handlers: dict[str, LocalToolHandler] | None = None,
  tool_definitions: list[dict[str, Any]] | None = None,
  skills_dir: str | Path | None = None,
  skills_excluded_tools: set[str] | None = None,
  skill_state_file: str | Path | None = None,
  outputs_dir: str | Path | None = None,
  code_execution: bool = False,
  code_execution_config: CodeExecutionConfig | None = None,
  needs_approval: Callable[..., bool] | None = None,
  session_cache_denied_tools: frozenset[str] | None = None,
  max_turns: int | None = None,
  max_budget_usd: float | None = None,
  per_turn_timeout: int = 300,
  valid_api_keys: set[str] | None = None,
  jwt_secret: str | None = None,
  session_ttl: int = 3600,
  cors_origins: list[str] | None = None,
  prefix: str = "/api",
  on_usage: Callable[[UsageEvent], Awaitable[Any] | Any] | None = None,
  on_session_summary: Callable[[SessionUsageSummary], Awaitable[Any] | Any] | None = None,
  on_tool_result: Callable[[ToolResultContext], Awaitable[Any] | Any] | None = None,
  on_tool_timing: Callable[..., None] | None = None,
  on_startup: Callable[..., Any] | None = None,
  on_shutdown: Callable[..., Any] | None = None,
  credentials_resolver: CredentialsResolver | None = None,
  credentials_refresh_resolver: CredentialsRefreshResolver | None = None,
  resolver_timeout_seconds: float = 5.0,
  usage_ledger: UsageLedger | None = None,
  usage_ledger_dlq_path: Path | None = None,
  coordinator: CoordinatorConfig | None = None,
  commercial_usage_producer_factory: Callable[..., Awaitable[Any] | Any] | None = None,
  commercial_usage_shipper: Any | None = None,
  commercial_usage_reconciliation_shipper: Any | None = None,
  commercial_work_start_gate: CommercialWorkStartGate | None = None,
) -> FastAPI:
  """Create a ready-to-run FastAPI gateway backed by a `ModelProvider`.

  This is the shortest path from "I have a system prompt" to a streaming agent
  server. The helper wires together:

  - `GatewayServerConfig`
  - `ChatRuntime`
  - `AgentRunner`
  - optional MCP tool discovery
  - optional local Python tools
  - optional code execution
  - optional skills and `run_agent`

  By default it uses Anthropic, but you can switch to the built-in OpenAI
  adapter with `provider="openai"`, the ChatGPT Codex backend with
  `provider="codex"`, or pass a `ModelProvider` instance directly. Use
  `create_gateway_app()` when you need channel-specific runtimes, approval
  rules for arbitrary tools, or direct control over runner construction.

  Args:
    system_prompt: Prompt string, or a list of `(text, should_cache)` blocks for
      providers that support prompt caching.
    provider: Provider name (`"anthropic"`, `"codex"`, or `"openai"`) or a
      `ModelProvider` instance.
    model: Default model for incoming chat requests. When omitted, string
      providers use their provider-specific default model. Required when you
      pass a `ModelProvider` instance.
    provider_config: Provider-specific config merged into the auth config last.
      Use this for fields like `base_url` or `compat`.
    api_key: Provider API key. Anthropic falls back to `ANTHROPIC_API_KEY`;
      other providers read their own env vars internally when supported.
    auth_token: OAuth bearer token. Anthropic falls back to
      `ANTHROPIC_AUTH_TOKEN`; `provider="openai"` and `provider="codex"` use
      it as bearer auth.
    rates_file: Optional JSON rate-table override used for Anthropic provider
      construction. When omitted, the helper uses env/default resolution.
    max_tokens: Per-turn output token cap passed to the provider.
    mcp_servers: Inline MCP server definitions. When provided, the helper uses
      `McpClientManager(config_path=None, inline_servers=...)`.
    mcp_config_path: Alternate Claude desktop config file to read MCP servers
      from. When omitted, `MCP_CONFIG_PATH` is used if set; otherwise inline
      `mcp_servers` remain inline-only and file-backed MCP config uses the
      `McpClientManager` default of `~/.claude.json`.
    mcp_timeout_overrides: Optional per-server MCP tool timeout overrides in
      seconds.
    mcp_default_tool_timeout: Default MCP tool timeout in seconds.
    mcp_session_inject_servers: MCP servers that should receive `_session_id`
      injected into tool inputs at dispatch time.
    mcp_strip_input_fields: Input schema fields removed from advertised MCP
      tool schemas after discovery.
    tool_handlers: Local async Python tool handlers keyed by tool name.
    tool_definitions: Tool schemas exposed to the model for local handlers.
    skills_dir: Directory of markdown skill files. When set, `run_agent` is
      registered automatically unless you override it yourself.
    skills_excluded_tools: Tool names hidden from spawned sub-agents.
    skill_state_file: Optional JSON file used by callable skills with
      `persist_state: true`. Previous state is injected into the sub-agent
      prompt and `## STATE_UPDATE_JSON` updates from the final response are
      merged back into this file.
    outputs_dir: Directory for named-skill output files. Stale same-day outputs are cleaned before background sub-agent launch.
    code_execution: Enable built-in `code_execute` and `code_execute_status`.
      Docker is preferred; subprocess is the fallback when registered.
    code_execution_config: Optional `CodeExecutionConfig` override.
    max_turns: Optional hard stop for the conversation loop.
    max_budget_usd: Optional estimated-cost budget across the session.
    per_turn_timeout: Maximum seconds allowed for a single model turn.
    valid_api_keys: API keys accepted by `/chat/init`. If empty, any non-empty
      key is accepted.
    jwt_secret: JWT signing secret for session tokens. A random secret is
      generated when omitted.
    session_ttl: Session lifetime in seconds.
    cors_origins: Allowed CORS origins. Defaults to `["*"]`.
    prefix: Route prefix. Defaults to `/api`.
    on_usage: Optional usage hook invoked after each completed stream turn.
    on_session_summary: Optional hook invoked once with aggregate session usage.
    on_tool_result: Optional hook invoked after each tool result is prepared.
    on_tool_timing: Optional hook invoked after each tool finishes.
    on_startup: Optional async or sync startup callback.
    on_shutdown: Optional async or sync shutdown callback.
    credentials_resolver: Optional init-time resolver for per-user credentials.
    credentials_refresh_resolver: Optional stream-time resolver for rotating
      resolver-backed credentials after provider rate-limit, billing, or auth failures.
    commercial_work_start_gate: Default-off verifier and durable one-time
      consumption gate run before request-scoped runtime construction.

  Returns:
    A configured FastAPI application exposing session init, chat SSE streaming,
    tool approval, and health-check routes.

  Example:
    Minimal server:

    ```python
    from agent_gateway import create_agent

    app = create_agent("You are a concise assistant.")
    ```

    Switch providers:

    ```python
    from agent_gateway import create_agent

    app = create_agent(
      "You are a concise assistant.",
      provider="openai",
    )
    ```

    Add a local tool:

    ```python
    from agent_gateway import create_agent


    async def read_status(_tool_input, **_kwargs):
      return {"status": "green"}, None


    app = create_agent(
      "Use read_status when the user asks for deployment state.",
      tool_handlers={"read_status": read_status},
      tool_definitions=[
        {
          "name": "read_status",
          "description": "Read the current deployment status.",
          "input_schema": {"type": "object", "properties": {}},
        }
      ],
    )
    ```
  """
  skill_state_store = SkillStateStore(skill_state_file) if skill_state_file is not None else None
  if credentials_resolver is None:
    provider_instance, _provider_name, auth_config = _resolve_provider(
      provider,
      model,
      api_key,
      auth_token,
      provider_config,
      max_tokens=max_tokens,
    )
  else:
    provider_instance, _provider_name, auth_config = _resolve_provider(
      provider,
      model,
      None,
      None,
      provider_config,
      auth_config={},
      max_tokens=max_tokens,
  )
  if isinstance(provider, str) and provider.strip().lower() == "anthropic":
    provider_instance = AnthropicProvider(rate_table=load_rate_table(rates_file))
  model = str(auth_config.get("model") or _get_default_model_for_provider(_provider_name))

  skill_loader = SkillLoader(skills_dir) if skills_dir else None
  mcp_client: McpClientManager | None = None
  resolved_mcp_config_path = (
    mcp_config_path if mcp_config_path is not None else _mcp_config_path_from_env()
  )
  if mcp_servers or resolved_mcp_config_path:
    builtin_names = set((tool_handlers or {}).keys())
    if code_execution:
      builtin_names |= {"code_execute", "code_execute_status"}
    if skills_dir and "run_agent" not in builtin_names:
      builtin_names |= {"run_agent", "get_background_result", "send_message"}
    mcp_client = McpClientManager(
      inline_servers=mcp_servers,
      config_path=resolved_mcp_config_path,
      builtin_tool_names=builtin_names,
      timeout_overrides=mcp_timeout_overrides,
      default_tool_timeout=mcp_default_tool_timeout,
      strip_input_fields=mcp_strip_input_fields,
    )

  app_ref: list[FastAPI | None] = [None]
  shipper_stop = asyncio.Event()
  shipper_task: list[asyncio.Task | None] = [None]
  reconciliation_shipper_task: list[asyncio.Task | None] = [None]

  def _report_shipper_exit(task: asyncio.Task) -> None:
    if task.cancelled():
      return
    error = task.exception()
    if error is not None:
      log.critical("commercial usage shipper exited unexpectedly: %s", error)
    elif not shipper_stop.is_set():
      log.critical("commercial usage shipper exited before shutdown")

  def _report_reconciliation_shipper_exit(task: asyncio.Task) -> None:
    if task.cancelled():
      return
    error = task.exception()
    if error is not None:
      log.critical("commercial reconciliation shipper exited unexpectedly: %s", error)
    elif not shipper_stop.is_set():
      log.critical("commercial reconciliation shipper exited before shutdown")

  async def _build_chat_runtime(
    session: GatewaySession,
    request: ChatRequest,
    channel: str | None,
    auth_manager: AuthManager,
  ) -> ChatRuntime:
    _ = auth_manager
    commercial_usage_producer = None
    if commercial_usage_producer_factory is not None:
      commercial_usage_producer = _call_commercial_usage_producer_factory(
        commercial_usage_producer_factory,
        session=session,
        request=request,
        channel=channel,
      )
      if inspect.isawaitable(commercial_usage_producer):
        commercial_usage_producer = await commercial_usage_producer
    local_handlers = dict(tool_handlers or {})
    extra_tool_defs = list(tool_definitions or [])
    runner_ref: list[Any] = [None]
    user_needs_approval = needs_approval

    ce_bundle = None
    if code_execution:
      ce_bundle = build_code_execution(session, config=code_execution_config)
      local_handlers.update(ce_bundle.handlers)
      extra_tool_defs.extend(ce_bundle.tool_definitions)

    if skill_loader is not None and "run_agent" not in local_handlers:
      from .sub_agent import (
        make_get_background_result_handler,
        make_get_background_result_tool_def,
        make_send_message_handler,
        make_send_message_tool_def,
        make_run_agent_handler,
        make_run_agent_tool_def,
      )

      local_handlers["run_agent"] = make_run_agent_handler(
        runner_ref,
        parent_session=session,
        skill_loader=skill_loader,
        mcp_client=mcp_client or _NullMcpClient(),
        needs_approval=user_needs_approval,
        mcp_session_inject_servers=mcp_session_inject_servers,
        local_tool_handlers=local_handlers,
        excluded_tools=skills_excluded_tools,
        skill_state_store=skill_state_store,
        outputs_dir=Path(outputs_dir) if outputs_dir is not None else None,
        default_model=model,
        allowed_models=allowed_models,
        coordinator_config=coordinator,
        approval_key_qualifier=ce_bundle.approval_qualifier if ce_bundle else None,
      )
      if "get_background_result" not in local_handlers:
        local_handlers["get_background_result"] = make_get_background_result_handler(runner_ref)
      if "send_message" not in local_handlers:
        local_handlers["send_message"] = make_send_message_handler(runner_ref)
      if not any(definition.get("name") == "run_agent" for definition in extra_tool_defs):
        extra_tool_defs.append(make_run_agent_tool_def(skill_loader))
      if not any(definition.get("name") == "get_background_result" for definition in extra_tool_defs):
        extra_tool_defs.append(make_get_background_result_tool_def())
      if not any(definition.get("name") == "send_message" for definition in extra_tool_defs):
        extra_tool_defs.append(make_send_message_tool_def())

    def _get_tool_defs() -> list[dict[str, Any]]:
      defs: list[dict[str, Any]] = []
      if mcp_client is not None:
        defs.extend(mcp_client.get_tool_definitions())
      defs.extend(extra_tool_defs)
      return defs

    approval_qualifier = ce_bundle.approval_qualifier if ce_bundle is not None else None

    def _needs_approval(
      name: str,
      tool_input: dict[str, Any] | None = None,
      qualifier: str = "",
    ) -> bool:
      if ce_bundle is not None and name == "code_execute":
        return ce_bundle.needs_approval(name, tool_input, qualifier)
      if user_needs_approval is not None:
        return bool(user_needs_approval(name, tool_input, qualifier))
      return False

    async def _combined_on_tool_result(ctx: ToolResultContext):
      if ce_bundle is not None:
        ce_bundle.sanitize_hook(ctx)
      if on_tool_result is None:
        return None
      result = on_tool_result(ctx)
      if inspect.isawaitable(result):
        return await result
      return result

    resolved_billing_mode = str((session.auth_config or auth_config).get("billing_mode", "byok"))
    rate_table = getattr(provider_instance, "_rate_table", None)
    resolved_rate_table_version = str(
      (session.auth_config or auth_config).get("rate_table_version")
      or getattr(rate_table, "version", "unknown")
      or "unknown"
    )
    resolved_usage_dlq_path = (
      Path(usage_ledger_dlq_path).expanduser() if usage_ledger_dlq_path is not None else DEFAULT_USAGE_DLQ_PATH
    )

    async def _combined_on_usage(event: UsageEvent) -> None:
      if usage_ledger is not None:
        await usage_ledger.record(event)
      if on_usage is not None:
        try:
          result = on_usage(event)
          if inspect.isawaitable(result):
            await result
        except Exception as exc:
          if usage_ledger is None:
            raise
          log.warning("on_usage observer failed after ledger record (non-fatal): %s", exc)

    async def _combined_on_session_summary(summary: SessionUsageSummary) -> None:
      session.cached_usage = summary
      if on_session_summary is not None:
        result = on_session_summary(summary)
        if inspect.isawaitable(result):
          await result

    def _build_runner(event_log, session_id: str, started_at: float | None = None) -> AgentRunner:
      resolved_started_at = float(started_at if started_at is not None else session.created_at)
      dispatcher = ToolDispatcher(
        mcp_client=mcp_client or _NullMcpClient(),
        local_tool_handlers=local_handlers,
        needs_approval=_needs_approval,
        request_approval=_make_request_approval(session, event_log),
        approved_tool_types=session.approved_tool_types,
        event_log=event_log,
        session_id=session_id,
        mcp_session_inject_servers=mcp_session_inject_servers,
        approval_key_qualifier=approval_qualifier,
        session_cache_denied_tools=session_cache_denied_tools,
        get_tool_definitions=_get_tool_defs,
      )
      runner = AgentRunner(
        event_log=event_log,
        dispatcher=dispatcher,
        session_id=session_id,
        started_at=resolved_started_at,
        provider=provider_instance,
        auth_config=session.auth_config or auth_config,
        mcp_client=mcp_client,
        get_tool_definitions=_get_tool_defs,
        per_turn_timeout=per_turn_timeout,
        on_tool_result=_combined_on_tool_result,
        on_usage=_combined_on_usage if usage_ledger is not None or on_usage is not None else None,
        on_session_summary=_combined_on_session_summary,
        on_tool_timing=on_tool_timing,
        user_id=session.user_id,
        request_id=request.request_id,
        billing_mode=resolved_billing_mode,
        rate_table_version=resolved_rate_table_version,
        channel=channel,
        usage_ledger_dlq_path=resolved_usage_dlq_path,
        max_budget_usd=max_budget_usd,
        coordinator=coordinator,
        code_execution_spill_dir_provider=ce_bundle.ensure_work_dir if ce_bundle else None,
        commercial_usage_producer=commercial_usage_producer,
      )
      runner_ref[0] = runner
      return runner

    return ChatRuntime(
      system_prompt=system_prompt,
      build_runner=_build_runner,
      get_tool_definitions=_get_tool_defs,
      provider=provider_instance,
      model_override=request.model or model,
      max_turns=max_turns,
    )

  async def _combined_startup() -> None:
    if mcp_client is not None:
      await mcp_client.startup()
    if on_startup is not None:
      result = on_startup()
      if inspect.isawaitable(result):
        await result
    if commercial_usage_shipper is not None and shipper_task[0] is None:
      shipper_stop.clear()
      shipper_task[0] = asyncio.create_task(
        commercial_usage_shipper.run_forever(shipper_stop),
        name="commercial-usage-shipper",
      )
      shipper_task[0].add_done_callback(_report_shipper_exit)
    if (
      commercial_usage_reconciliation_shipper is not None
      and reconciliation_shipper_task[0] is None
    ):
      shipper_stop.clear()
      reconciliation_shipper_task[0] = asyncio.create_task(
        commercial_usage_reconciliation_shipper.run_forever(shipper_stop),
        name="commercial-usage-reconciliation-shipper",
      )
      reconciliation_shipper_task[0].add_done_callback(
        _report_reconciliation_shipper_exit
      )

  async def _combined_shutdown() -> None:
    if shipper_task[0] is not None:
      shipper_stop.set()
      try:
        await shipper_task[0]
      except Exception as exc:
        log.error("commercial usage shipper stopped with an error: %s", exc)
      shipper_task[0] = None
    if reconciliation_shipper_task[0] is not None:
      shipper_stop.set()
      try:
        await reconciliation_shipper_task[0]
      except Exception as exc:
        log.error("commercial reconciliation shipper stopped with an error: %s", exc)
      reconciliation_shipper_task[0] = None
    if code_execution and app_ref[0] is not None:
      from .code_execution import cleanup_code_execution

      session_store = app_ref[0].state.auth.session_store
      for session in list(session_store.sessions.values()):
        await cleanup_code_execution(session)
    if on_shutdown is not None:
      result = on_shutdown()
      if inspect.isawaitable(result):
        await result
    if mcp_client is not None:
      await mcp_client.shutdown()

  allowed_models: set[str] = set()

  app = create_gateway_app(
    GatewayServerConfig(
      jwt_secret=jwt_secret or secrets.token_hex(32),
      valid_api_keys=valid_api_keys or set(),
      session_ttl=session_ttl,
      cors_origins=["*"] if cors_origins is None else cors_origins,
      allowed_models=allowed_models,
      build_chat_runtime=_build_chat_runtime,
      default_provider=provider_instance,
      auth_config=auth_config,
      credentials_resolver=credentials_resolver,
      credentials_refresh_resolver=credentials_refresh_resolver,
      resolver_timeout_seconds=resolver_timeout_seconds,
      commercial_work_start_gate=commercial_work_start_gate,
      mcp_client=mcp_client,
      per_turn_timeout=per_turn_timeout,
      on_startup=_combined_startup,
      on_shutdown=_combined_shutdown,
      prefix=prefix,
    )
  )
  app_ref[0] = app

  if code_execution:
    from .code_execution import cleanup_code_execution

    session_store = app.state.auth.session_store
    session_store.add_on_expiry(cleanup_code_execution)

  if mcp_session_inject_servers and mcp_client is not None:
    session_store = app.state.auth.session_store

    async def _on_expiry_with_mcp_session_cleanup(session: GatewaySession) -> None:
      for server_name in mcp_session_inject_servers:
        close_tool = mcp_client.resolve_tool_name(server_name, f"{server_name}_close_session")
        if close_tool:
          _result, err = await mcp_client.call_tool(close_tool, {"_session_id": session.session_id})
          if err:
            log.warning(
              "MCP session cleanup failed for %s/%s: %s",
              server_name,
              session.session_id,
              err.get("message", ""),
            )
        else:
          log.debug("No close_session tool for %s; session cleanup skipped", server_name)

    session_store.add_on_expiry(_on_expiry_with_mcp_session_cleanup)

  return app


__all__ = ["create_agent"]
