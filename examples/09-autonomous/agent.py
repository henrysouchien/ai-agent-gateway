import os
from pathlib import Path

from agent_gateway import (
  BoundCapabilityExecution,
  CapabilityBind,
  CredentialHandle,
  DeliveryConfig,
  GatewaySession,
  INITIAL_MODEL_REGISTRY,
  INITIAL_MODEL_SELECTION_POLICY,
  run_autonomous_sync,
)
from agent_gateway.providers import AnthropicProvider


BASE_DIR = Path(__file__).parent
STATE_DIR = BASE_DIR / "state"


async def read_status(_tool_input, **_kwargs):
  return {
    "service": "agent-gateway",
    "status": "green",
    "checked_by": "09-autonomous example",
  }, None


TOOL_DEFINITIONS = [
  {
    "name": "read_status",
    "description": "Read the current service status before sending the final summary.",
    "input_schema": {
      "type": "object",
      "properties": {},
    },
  },
]


if __name__ == "__main__":
  api_key = os.getenv("ANTHROPIC_API_KEY", "").strip()
  if not api_key:
    raise RuntimeError("Set ANTHROPIC_API_KEY before running this example.")

  # Standalone fixture only: production callers should obtain this immutable
  # bind and credential snapshot from their server-owned capability resolver.
  entry = INITIAL_MODEL_REGISTRY.require("anthropic.claude-opus-5")
  credential_handle = CredentialHandle(
    handle_id="example-09-anthropic-service",
    provider=entry.provider,
    principal="service",
    tenant_id="example-09-autonomous",
    actor_id=None,
  )
  capability_bind = CapabilityBind(
    schema_version="1.0",
    capability_id="session.driver",
    model_key=entry.key,
    provider=entry.provider,
    upstream_model=entry.upstream_model,
    adapter=entry.adapter,
    protocol_profile=entry.protocol_profile,
    route=entry.route,
    effort="high",
    credential_principal="service",
    credential_ref=credential_handle.handle_id,
    run_mode="cron",
    registry_revision=INITIAL_MODEL_REGISTRY.revision,
    policy_revision=INITIAL_MODEL_SELECTION_POLICY.revision,
    selection_source="capability_default",
  )
  capability_execution = BoundCapabilityExecution(
    bind=capability_bind,
    registry=INITIAL_MODEL_REGISTRY,
    adapter=AnthropicProvider(),
    auth_config={
      "provider": capability_bind.provider,
      "max_tokens": 16_000,
      "auth_mode": "api",
      "api_key": api_key,
      "credential_handle_id": credential_handle.handle_id,
      "tenant_id": credential_handle.tenant_id,
    },
  )

  result = run_autonomous_sync(
    "You are a concise operations assistant. Always call read_status before you reply.",
    "Check the current service status and send a short summary.",
    capability_execution=capability_execution,
    session=GatewaySession(
      session_id="autonomous-example",
      api_key_hash="example",
      created_at=0,
      expires_at=2**63 - 1,
      user_id="autonomous-example",
      role="owner",
      capabilities=frozenset({"session.driver"}),
      model_entitled_capabilities=frozenset({"session.driver"}),
      model_entitled_keys=frozenset({entry.key}),
      tenant_id=credential_handle.tenant_id,
      session_credential_handle=credential_handle,
      allow_service_for_interactive=True,
      channel="cli",
    ),
    tool_handlers={"read_status": read_status},
    tool_definitions=TOOL_DEFINITIONS,
    state_dir=STATE_DIR,
    delivery=DeliveryConfig(
      telegram_bot_token=os.getenv("TELEGRAM_BOT_TOKEN", "").strip() or None,
      telegram_chat_id=os.getenv("TELEGRAM_CHAT_ID", "").strip() or None,
      telegram_label="Agent Gateway autonomous example",
    ),
    user_id="autonomous-example",
    billing_mode="byok",
    rate_table_version="example",
  )
  print(result.response)
