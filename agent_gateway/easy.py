from __future__ import annotations

import logging
import inspect
import secrets
from pathlib import Path
from typing import Any, Awaitable, Callable

from fastapi import FastAPI

from ._provider_utils import _resolve_provider
from .auth import CredentialsResolver
from .code_execution import CodeExecutionConfig, build_code_execution
from .mcp_client import McpClientManager
from .multi_user.billing import DEFAULT_USAGE_DLQ_PATH, UsageEvent, UsageLedger
from .providers import AnthropicProvider, ModelProvider
from .rates import load_rate_table
from .runner import AgentRunner, ToolResultContext
from .server import ChatRequest, ChatRuntime, GatewayServerConfig, _make_request_approval, create_gateway_app
from .session import AuthManager, Session
from .skills import SkillLoader
from .tool_dispatcher import LocalToolHandler, ToolDispatcher

log = logging.getLogger("agent_gateway.easy")


class _NullMcpClient:
  def is_mcp_tool(self, _name: str) -> bool:
    return False

  async def call_tool(self, name: str, _tool_input: dict[str, Any]):
    return None, {"code": "unknown_tool", "message": f"Unknown tool: {name}"}

  def get_tool_definitions(self) -> list[dict[str, Any]]:
    return []


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
  outputs_dir: str | Path | None = None,
  code_execution: bool = False,
  code_execution_config: CodeExecutionConfig | None = None,
  max_turns: int | None = None,
  max_budget_usd: float | None = None,
  per_turn_timeout: int = 300,
  valid_api_keys: set[str] | None = None,
  jwt_secret: str | None = None,
  session_ttl: int = 3600,
  cors_origins: list[str] | None = None,
  prefix: str = "/api",
  on_usage: Callable[[UsageEvent], Awaitable[Any] | Any] | None = None,
  on_tool_result: Callable[[ToolResultContext], Awaitable[Any] | Any] | None = None,
  on_tool_timing: Callable[..., None] | None = None,
  on_startup: Callable[..., Any] | None = None,
  on_shutdown: Callable[..., Any] | None = None,
  credentials_resolver: CredentialsResolver | None = None,
  resolver_timeout_seconds: float = 5.0,
  usage_ledger: UsageLedger | None = None,
  usage_ledger_dlq_path: Path | None = None,
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
      from. Defaults to `~/.claude.json` when omitted and `mcp_servers` is not
      provided.
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
    on_usage: Optional usage hook invoked after each completed stream.
    on_tool_result: Optional hook invoked after each tool result is prepared.
    on_tool_timing: Optional hook invoked after each tool finishes.
    on_startup: Optional async or sync startup callback.
    on_shutdown: Optional async or sync shutdown callback.

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
  model = str(auth_config["model"])

  skill_loader = SkillLoader(skills_dir) if skills_dir else None
  mcp_client: McpClientManager | None = None
  if mcp_servers or mcp_config_path:
    builtin_names = set((tool_handlers or {}).keys())
    if code_execution:
      builtin_names |= {"code_execute", "code_execute_status"}
    if skills_dir and "run_agent" not in builtin_names:
      builtin_names |= {"run_agent", "get_background_result"}
    mcp_client = McpClientManager(
      inline_servers=mcp_servers,
      config_path=mcp_config_path,
      builtin_tool_names=builtin_names,
      timeout_overrides=mcp_timeout_overrides,
      default_tool_timeout=mcp_default_tool_timeout,
      strip_input_fields=mcp_strip_input_fields,
    )

  app_ref: list[FastAPI | None] = [None]

  async def _build_chat_runtime(
    session: Session,
    request: ChatRequest,
    channel: str | None,
    auth_manager: AuthManager,
  ) -> ChatRuntime:
    _ = auth_manager
    local_handlers = dict(tool_handlers or {})
    extra_tool_defs = list(tool_definitions or [])
    runner_ref: list[Any] = [None]

    ce_bundle = None
    if code_execution:
      ce_bundle = build_code_execution(session, config=code_execution_config)
      local_handlers.update(ce_bundle.handlers)
      extra_tool_defs.extend(ce_bundle.tool_definitions)

    if skill_loader is not None and "run_agent" not in local_handlers:
      from .sub_agent import (
        make_get_background_result_handler,
        make_get_background_result_tool_def,
        make_run_agent_handler,
        make_run_agent_tool_def,
      )

      local_handlers["run_agent"] = make_run_agent_handler(
        runner_ref,
        parent_session=session,
        skill_loader=skill_loader,
        mcp_client=mcp_client or _NullMcpClient(),
        mcp_session_inject_servers=mcp_session_inject_servers,
        local_tool_handlers=local_handlers,
        excluded_tools=skills_excluded_tools,
        outputs_dir=Path(outputs_dir) if outputs_dir is not None else None,
        default_model=model,
        allowed_models=allowed_models,
      )
      if "get_background_result" not in local_handlers:
        local_handlers["get_background_result"] = make_get_background_result_handler(runner_ref)
      if not any(definition.get("name") == "run_agent" for definition in extra_tool_defs):
        extra_tool_defs.append(make_run_agent_tool_def(skill_loader))
      if not any(definition.get("name") == "get_background_result" for definition in extra_tool_defs):
        extra_tool_defs.append(make_get_background_result_tool_def())

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

    def _build_runner(event_log, session_id: str) -> AgentRunner:
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
      )
      runner = AgentRunner(
        event_log=event_log,
        dispatcher=dispatcher,
        session_id=session_id,
        provider=provider_instance,
        auth_config=session.auth_config or auth_config,
        mcp_client=mcp_client,
        get_tool_definitions=_get_tool_defs,
        per_turn_timeout=per_turn_timeout,
        on_tool_result=_combined_on_tool_result,
        on_usage=_combined_on_usage if usage_ledger is not None or on_usage is not None else None,
        on_tool_timing=on_tool_timing,
        user_id=session.user_id,
        request_id=request.request_id,
        billing_mode=resolved_billing_mode,
        rate_table_version=resolved_rate_table_version,
        channel=channel,
        usage_ledger_dlq_path=resolved_usage_dlq_path,
        max_budget_usd=max_budget_usd,
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

  async def _combined_shutdown() -> None:
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
      resolver_timeout_seconds=resolver_timeout_seconds,
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
    code_execution_expiry = session_store._on_expiry

    async def _on_expiry_with_cleanup(session: Session) -> None:
      if code_execution_expiry is not None:
        result = code_execution_expiry(session)
        if inspect.isawaitable(result):
          await result
      await cleanup_code_execution(session)

    session_store.set_on_expiry(_on_expiry_with_cleanup)

  if mcp_session_inject_servers and mcp_client is not None:
    session_store = app.state.auth.session_store
    mcp_session_expiry = session_store._on_expiry

    async def _on_expiry_with_mcp_session_cleanup(session: Session) -> None:
      if mcp_session_expiry is not None:
        result = mcp_session_expiry(session)
        if inspect.isawaitable(result):
          await result
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

    session_store.set_on_expiry(_on_expiry_with_mcp_session_cleanup)

  return app


__all__ = ["create_agent"]
