import os

from claude_gateway import (
  AgentRunner,
  ChatRuntime,
  GatewayServerConfig,
  McpClientManager,
  OpenAIProvider,
  ToolDispatcher,
  create_gateway_app,
)


DEFAULT_MODEL = "gpt-4o-mini"
ALLOWED_MODELS = {DEFAULT_MODEL, "gpt-4o", "o3-mini", "o1-mini", "o1"}


def build_auth_config() -> dict[str, object]:
  base_url = os.getenv("OPENAI_BASE_URL", "").strip()
  config: dict[str, object] = {
    "api_key": os.getenv("OPENAI_API_KEY", "").strip(),
    "model": DEFAULT_MODEL,
    "max_tokens": 8_000,
  }
  if base_url:
    config["base_url"] = base_url
  return config


provider = OpenAIProvider()
mcp_client = McpClientManager(config_path=None)
AUTH_CONFIG = build_auth_config()


async def build_chat_runtime(session, request, channel, auth_manager) -> ChatRuntime:
  _ = session, channel, auth_manager

  def get_tool_definitions():
    return []

  def build_runner(event_log, session_id):
    dispatcher = ToolDispatcher(
      mcp_client=mcp_client,
      local_tool_handlers={},
      session_id=session_id,
    )
    return AgentRunner(
      event_log=event_log,
      dispatcher=dispatcher,
      session_id=session_id,
      provider=provider,
      auth_config={
        **AUTH_CONFIG,
        "model": request.model or DEFAULT_MODEL,
      },
      mcp_client=mcp_client,
      get_tool_definitions=get_tool_definitions,
    )

  return ChatRuntime(
    system_prompt="You are a concise assistant running through the OpenAI provider adapter.",
    build_runner=build_runner,
    get_tool_definitions=get_tool_definitions,
    provider=provider,
    model_override=request.model or DEFAULT_MODEL,
    max_turns=6,
  )


app = create_gateway_app(
  GatewayServerConfig(
    valid_api_keys={"demo-key"},
    cors_origins=["*"],
    allowed_models=ALLOWED_MODELS,
    build_chat_runtime=build_chat_runtime,
    default_provider=provider,
    auth_config=AUTH_CONFIG,
    mcp_client=mcp_client,
    on_startup=mcp_client.startup,
    on_shutdown=mcp_client.shutdown,
  )
)
