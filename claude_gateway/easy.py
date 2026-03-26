from __future__ import annotations

import inspect
import os
import secrets
from pathlib import Path
from typing import Any, Awaitable, Callable

from fastapi import FastAPI

from .code_execution import CodeExecutionConfig, build_code_execution
from .mcp_client import McpClientManager
from .providers import AnthropicProvider
from .runner import AgentRunner, ToolResultContext
from .server import ChatRequest, ChatRuntime, GatewayServerConfig, _make_request_approval, create_gateway_app
from .session import AuthManager, Session
from .skills import SkillLoader
from .tool_dispatcher import LocalToolHandler, ToolDispatcher


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
  model: str = "claude-sonnet-4-6",
  api_key: str | None = None,
  auth_token: str | None = None,
  max_tokens: int = 16_000,
  mcp_servers: dict[str, dict[str, Any]] | None = None,
  mcp_config_path: str | Path | None = None,
  tool_handlers: dict[str, LocalToolHandler] | None = None,
  tool_definitions: list[dict[str, Any]] | None = None,
  skills_dir: str | Path | None = None,
  skills_excluded_tools: set[str] | None = None,
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
  on_usage: Callable[[dict[str, Any]], None] | None = None,
  on_tool_result: Callable[[ToolResultContext], Awaitable[Any] | Any] | None = None,
  on_tool_timing: Callable[..., None] | None = None,
  on_startup: Callable[..., Any] | None = None,
  on_shutdown: Callable[..., Any] | None = None,
) -> FastAPI:
  """Create a ready-to-run FastAPI gateway backed by `AnthropicProvider`.

  This is the shortest path from "I have a system prompt" to a streaming agent
  server. The helper wires together:

  - `GatewayServerConfig`
  - `ChatRuntime`
  - `AgentRunner`
  - optional MCP tool discovery
  - optional local Python tools
  - optional code execution
  - optional skills and `run_agent`

  `create_agent()` currently supports Anthropic only. If you need OpenAI,
  channel-specific runtimes, approval rules for arbitrary tools, or direct
  control over runner construction, use `create_gateway_app()` instead.

  Args:
    system_prompt: Prompt string, or a list of `(text, should_cache)` blocks for
      providers that support prompt caching.
    model: Default Anthropic model for incoming chat requests.
    api_key: Anthropic API key. Falls back to `ANTHROPIC_API_KEY`.
    auth_token: Anthropic OAuth token. Falls back to `ANTHROPIC_AUTH_TOKEN`.
    max_tokens: Per-turn output token cap passed to the provider.
    mcp_servers: Inline MCP server definitions. When provided, the helper uses
      `McpClientManager(config_path=None, inline_servers=...)`.
    mcp_config_path: Alternate Claude desktop config file to read MCP servers
      from. Defaults to `~/.claude.json` when omitted and `mcp_servers` is not
      provided.
    tool_handlers: Local async Python tool handlers keyed by tool name.
    tool_definitions: Tool schemas exposed to the model for local handlers.
    skills_dir: Directory of markdown skill files. When set, `run_agent` is
      registered automatically unless you override it yourself.
    skills_excluded_tools: Tool names hidden from spawned sub-agents.
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
    from claude_gateway import create_agent

    app = create_agent("You are a concise assistant.")
    ```

    Add a local tool:

    ```python
    from claude_gateway import create_agent


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
  resolved_key = (api_key or "").strip() or os.environ.get("ANTHROPIC_API_KEY", "").strip()
  resolved_token = (auth_token or "").strip() or os.environ.get("ANTHROPIC_AUTH_TOKEN", "").strip()

  if resolved_key:
    auth_config: dict[str, Any] = {
      "auth_mode": "api",
      "api_key": resolved_key,
      "auth_token": "",
    }
  elif resolved_token:
    auth_config = {
      "auth_mode": "oauth",
      "api_key": "",
      "auth_token": resolved_token,
    }
  else:
    auth_config = {
      "auth_mode": "api",
      "api_key": "",
      "auth_token": "",
    }

  auth_config["model"] = model
  auth_config["max_tokens"] = max_tokens

  skill_loader = SkillLoader(skills_dir) if skills_dir else None
  mcp_client: McpClientManager | None = None
  if mcp_servers or mcp_config_path:
    builtin_names = set((tool_handlers or {}).keys())
    if code_execution:
      builtin_names |= {"code_execute", "code_execute_status"}
    if skills_dir:
      builtin_names.add("run_agent")
    mcp_client = McpClientManager(
      inline_servers=mcp_servers,
      config_path=mcp_config_path,
      builtin_tool_names=builtin_names,
    )

  provider = AnthropicProvider()
  app_ref: list[FastAPI | None] = [None]

  async def _build_chat_runtime(
    session: Session,
    request: ChatRequest,
    channel: str | None,
    auth_manager: AuthManager,
  ) -> ChatRuntime:
    _ = channel, auth_manager
    local_handlers = dict(tool_handlers or {})
    extra_tool_defs = list(tool_definitions or [])
    runner_ref: list[Any] = [None]

    ce_bundle = None
    if code_execution:
      ce_bundle = build_code_execution(session, config=code_execution_config)
      local_handlers.update(ce_bundle.handlers)
      extra_tool_defs.extend(ce_bundle.tool_definitions)

    if skill_loader is not None and "run_agent" not in local_handlers:
      from .sub_agent import make_run_agent_handler, make_run_agent_tool_def

      local_handlers["run_agent"] = make_run_agent_handler(
        runner_ref,
        skill_loader=skill_loader,
        mcp_client=mcp_client or _NullMcpClient(),
        local_tool_handlers=local_handlers,
        excluded_tools=skills_excluded_tools,
        default_model=model,
        allowed_models=allowed_models,
      )
      if not any(definition.get("name") == "run_agent" for definition in extra_tool_defs):
        extra_tool_defs.append(make_run_agent_tool_def(skill_loader))

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

    def _build_runner(event_log, session_id: str) -> AgentRunner:
      dispatcher = ToolDispatcher(
        mcp_client=mcp_client or _NullMcpClient(),
        local_tool_handlers=local_handlers,
        needs_approval=_needs_approval,
        request_approval=_make_request_approval(session, event_log),
        approved_tool_types=session.approved_tool_types,
        event_log=event_log,
        session_id=session_id,
        approval_key_qualifier=approval_qualifier,
      )
      runner = AgentRunner(
        event_log=event_log,
        dispatcher=dispatcher,
        session_id=session_id,
        provider=provider,
        auth_config=auth_config,
        mcp_client=mcp_client,
        get_tool_definitions=_get_tool_defs,
        per_turn_timeout=per_turn_timeout,
        on_tool_result=_combined_on_tool_result,
        on_usage=on_usage,
        on_tool_timing=on_tool_timing,
        max_budget_usd=max_budget_usd,
      )
      runner_ref[0] = runner
      return runner

    return ChatRuntime(
      system_prompt=system_prompt,
      build_runner=_build_runner,
      get_tool_definitions=_get_tool_defs,
      provider=provider,
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

  allowed_models = {"claude-sonnet-4-6", "claude-opus-4-6"}
  allowed_models.add(model)

  app = create_gateway_app(
    GatewayServerConfig(
      jwt_secret=jwt_secret or secrets.token_hex(32),
      valid_api_keys=valid_api_keys or set(),
      session_ttl=session_ttl,
      cors_origins=["*"] if cors_origins is None else cors_origins,
      allowed_models=allowed_models,
      build_chat_runtime=_build_chat_runtime,
      default_provider=provider,
      auth_config=auth_config,
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
    existing_expiry = session_store._on_expiry

    async def _on_expiry_with_cleanup(session: Session) -> None:
      if existing_expiry is not None:
        result = existing_expiry(session)
        if inspect.isawaitable(result):
          await result
      await cleanup_code_execution(session)

    session_store.set_on_expiry(_on_expiry_with_cleanup)

  return app


__all__ = ["create_agent"]
