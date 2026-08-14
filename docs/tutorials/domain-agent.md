# Build A Domain Agent

Use `create_agent()` for simple agents. Use `create_gateway_app()` when the
agent is part of a larger product with its own auth, domain tools, storage,
approval rules, and background workflows.

This guide describes the consumer pattern used by domain systems such as a
portfolio research assistant.

## Runtime Shape

The product owns domain behavior. `ai-agent-gateway` owns the runtime shell.

```text
domain app
  auth, user profile, memory, domain tools
        |
        v
GatewayServerConfig
        |
        v
build_chat_runtime(session, request, channel, auth_manager)
        |
        v
ChatRuntime
        |
        v
AgentRunner + ToolDispatcher
```

Keep these boundaries explicit:

- `GatewayServerConfig` owns HTTP/session hooks plus the tenant-scoped
  product model registry, selection policy, credential handles, materializer,
  and adapter resolver.
- The gateway prepares and freezes the `session.driver` model, effort,
  credential principal, admitting registry, provider adapter, and credential
  snapshot before calling `build_chat_runtime`.
- `build_chat_runtime` consumes that exact prepared execution identity and
  chooses only domain behavior such as prompts, tools, limits, and hooks.
- `ChatRuntime` preserves the exact `BoundCapabilityExecution` object in the
  per-request wiring bundle returned to the gateway.
- `AgentRunner` runs the model/tool loop.
- `ToolDispatcher` is the policy boundary before any local or MCP tool executes.

## Canonical Custom Runtime

The complete server-owned setup is intentionally centralized rather than
repeated here. Start from the runnable
[full production gateway](../../examples/07-full-production/), which
defines the registry/policy, opaque service handle, materializer, and adapter
resolver. Its domain-specific runtime factory follows
this contract:

```python
from agent_gateway import AgentRunner, ChatRuntime, ToolDispatcher


async def build_chat_runtime(session, request, channel, auth_manager):
  _ = auth_manager
  capability_execution = request.capability_execution
  if capability_execution is None:
    raise RuntimeError("Runtime requires a prepared session.driver turn")
  capability_execution.validate()
  bound_auth_config = capability_execution.auth_config

  prompt = (
    "You are a domain assistant. Use tools for facts, cite tool outputs, "
    "and ask for approval before mutating external state."
  )

  def build_runner(event_log, session_id: str, started_at: float) -> AgentRunner:
    dispatcher = ToolDispatcher(
      mcp_client=mcp_client,
      local_tool_handlers=tool_handlers,
      event_log=event_log,
      session_id=session_id,
      user_id=session.user_id,
      channel=channel,
    )
    return AgentRunner(
      event_log=event_log,
      dispatcher=dispatcher,
      session_id=session_id,
      started_at=started_at,
      capability_execution=capability_execution,
      get_tool_definitions=mcp_client.get_tool_definitions,
      mcp_client=mcp_client,
      user_id=session.user_id,
      request_id=request.request_id,
      channel=channel,
      rate_table_version=str(bound_auth_config["rate_table_version"]),
      billing_mode=str(bound_auth_config["billing_mode"]),
    )

  return ChatRuntime(
    system_prompt=prompt,
    build_runner=build_runner,
    capability_execution=capability_execution,
    get_tool_definitions=mcp_client.get_tool_definitions,
    max_turns=8,
  )
```

The gateway calls `build_chat_runtime` only after canonical preparation
succeeds. Use request context there to select the domain profile, prompt, tool
set, and memory scope. Do not read `request.model_key` and reselect execution,
choose a provider, or merge raw credentials in the runtime factory. For custom
approval queue wiring, see
the runnable [tool-approval gateway](../../examples/06-tool-approval/).

## Add Domain Tools

Use MCP servers when tools are already separate processes or need their own
release cadence. Use local handlers for small in-process product operations.

```python
from typing import Any


async def read_position(input: dict[str, Any], **_: Any) -> tuple[dict[str, Any], None]:
  ticker = str(input["ticker"]).upper()
  return {"ticker": ticker, "weight": 0.04}, None


tool_handlers = {"read_position": read_position}
tool_definitions = [
  {
    "name": "read_position",
    "description": "Read the current portfolio weight for one ticker.",
    "input_schema": {
      "type": "object",
      "properties": {"ticker": {"type": "string"}},
      "required": ["ticker"],
    },
  }
]
```

Pass those into `ToolDispatcher(local_tool_handlers=tool_handlers)` and return
`tool_definitions + mcp_client.get_tool_definitions()` from `ChatRuntime`.

## Validate Tool Use

Put product policy in dispatch, not in prompt text alone.

Use `ToolDispatcher` controls for:

- `needs_approval`: request human approval before sensitive tools.
- `interceptors`: warn, ask, allow, or deny before execution.
- `allowed_mcp_tools_by_server`: narrow a server surface for one runtime.
- `session_cache_denied_tools`: avoid repeated asks for denied tool classes.
- `on_tool_result`: sanitize, summarize, or record structured tool metadata.

For example, a portfolio app can allow read-only market data tools while asking
for approval before order, notification, or artifact-write tools.

## Memory

The gateway includes generic memory primitives such as `MemoryStore` and
`MarkdownSyncManager`. A domain app should own the schema of its domain memory.

Recommended pattern:

1. Store durable product facts in product tables or files.
2. Mirror agent-readable summaries into `MemoryStore` or markdown sync.
3. Scope reads by `user_id`, tenant, portfolio, channel, or profile.
4. Treat memory writes as tool calls so they pass through policy and audit.

This keeps the gateway generic while giving the domain app control over
retention, privacy, and meaning.

## Autonomous Workflows

For cron or autonomous work, reuse the same runtime pieces:

- resolve a `session.driver` bind through server-owned policy
- materialize its credential handle into an immutable auth-config snapshot
- pass the exact admitting registry, bind, provider adapter, and credential
  snapshot
- configure the same domain prompts, MCP servers, and tool policy
- call `run_autonomous()` or `run_autonomous_sync()`
- deliver results through the product's notification layer

The key design rule is that background runs and interactive chats should share
domain validation, memory scoping, and tool approval semantics. Only delivery
and scheduling should differ.

## Production Checklist

- Configure one tenant-scoped product model registry and selection policy.
- Represent credentials with opaque handles and materialize only the selected
  handle through a trusted resolver.
- Set a credential materializer and adapter resolver.
- Make the product-owned `build_chat_runtime` callback consume and preserve the
  exact request bind, admitting registry, adapter, and credential snapshot.
- Keep tool schemas close to their handlers or MCP servers.
- Put mutation policy in dispatcher/interceptor code.
- Set explicit `user_id`, `channel`, `rate_table_version`, and `billing_mode`
  when using `AgentRunner` or `AgentSDKRunner`.
- Wire `on_usage`, `on_session_summary`, and `on_late_usage_event` before launch.
- Store transcripts and tool audit records in product-owned locations.
- Keep domain memory schema outside the gateway package.
- Use `create_agent()` only for simple deployments; use `create_gateway_app()`
  for product integrations.
