# ruff: noqa: E402

import inspect
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import pytest

ROOT = Path(__file__).resolve().parents[3]
PKG_DIR = ROOT / "packages" / "agent-gateway"
if str(PKG_DIR) not in sys.path:
  sys.path.insert(0, str(PKG_DIR))

from agent_gateway.event_log import EventLog
from agent_gateway import (
  CAPABILITY_IDS,
  CapabilityDefault,
  CapabilitySelectionPolicy,
  CredentialHandle,
  ModelRegistryEntry,
  ProductModelRegistry,
  ProductModelSelectionPolicy,
)
from agent_gateway.auth import AuthConfig, ResolverResult
from agent_gateway.providers.agent_sdk import AgentSDKConfig
from agent_gateway.providers import AnthropicProvider
from agent_gateway.runner import AgentRunner
from agent_gateway.sdk_runner import AgentSDKRunner
from agent_gateway.server import (
  ChatRuntime,
  GatewayServerConfig,
  MaterializedCredential,
  create_gateway_app,
)
from agent_gateway.tool_dispatcher import ToolDispatcher

class _NullMcpClient:
  def is_mcp_tool(self, _name: str) -> bool:
    return False

  async def call_tool(self, name: str, _tool_input: dict[str, Any]):
    return None, {"code": "unknown_tool", "message": f"Unknown tool: {name}"}

  def get_tool_definitions(self) -> list[dict[str, Any]]:
    return []

  def get_server_for_tool(self, _name: str) -> str | None:
    return None


@dataclass
class _TestAppState:
  runtime: ChatRuntime | None = None
  runner: Any = None
  event_log: EventLog | None = None
  session: Any = None
  disconnect_hook_calls: int = 0


@pytest.fixture
def auth_config_model_free() -> dict[str, Any]:
  return {"api_key": "test-key", "max_tokens": 256}


@pytest.fixture
def make_test_app():
  def _make_test_app(
    *,
    provider: Any = None,
    dispatcher: ToolDispatcher | Callable[[EventLog, str, Any], ToolDispatcher] | None = None,
    runner_class: type[AgentRunner] | type[AgentSDKRunner] = AgentRunner,
    runner_kwargs: dict[str, Any] | None = None,
    tool_definitions: list[dict[str, Any]] | None = None,
    auth_config: dict[str, Any] | None = None,
    on_disconnect: Callable[[_TestAppState], Any] | None = None,
    attach_disconnect_handler: bool = True,
    tool_call_timeout: float | None = 120.0,
    per_turn_timeout: float | None = 300.0,
    stream_stall_timeout: float | None = 60.0,
    sdk_config: AgentSDKConfig | None = None,
    transcript_dir: str | Path | None = None,
  ):
    state = _TestAppState()
    resolved_auth_config = dict(
      auth_config
      if auth_config is not None
      else {"api_key": "test-key", "max_tokens": 256}
    )
    resolved_tool_definitions = list(tool_definitions or [])
    resolved_sdk_config = sdk_config or AgentSDKConfig(
      user_id="alice",
      billing_mode="byok",
      rate_table_version="unknown",
    )
    execution_transport = (
      "agent-sdk"
      if issubclass(runner_class, AgentSDKRunner)
      else "native"
    )
    adapter_provider_name = (
      "anthropic"
      if execution_transport == "agent-sdk"
      else str(getattr(provider, "name", "anthropic") or "anthropic")
    )
    provider_name = (
      adapter_provider_name.strip().lower()
      if adapter_provider_name.strip().lower()
      in {"anthropic", "openai", "codex", "xai", "fixture"}
      else "anthropic"
    )
    if (
      execution_transport == "native"
      and provider is not None
      and adapter_provider_name.strip().lower() != provider_name
    ):
      provider.name = provider_name
    bound_model = "claude-sonnet-4-6"
    bound_effort = (
      "high"
      if execution_transport == "agent-sdk"
      else "none"
    )
    selection_fields = {
      "model",
      "model_key",
      "effort",
      "thinking",
      "thinking_enabled_requested",
    }
    bound_auth_config = {
      **{
        key: value
        for key, value in resolved_auth_config.items()
        if key not in selection_fields
      },
      "provider": provider_name,
      "auth_mode": str(
        resolved_auth_config.get("auth_mode") or "api"
      ).strip().lower(),
      "billing_mode": str(
        resolved_auth_config.get("billing_mode") or "byok"
      ),
      "rate_table_version": str(
        resolved_auth_config.get("rate_table_version") or "unknown"
      ),
    }
    model_key = "test.session-driver"
    adapter_id = (
      "anthropic.agent_sdk"
      if execution_transport == "agent-sdk"
      else "test.native"
    )
    registry_entry = ModelRegistryEntry(
      key=model_key,
      label="Gateway test model",
      provider=provider_name,
      upstream_model=bound_model,
      adapter=adapter_id,
      protocol_profile=(
        "agent_sdk.session"
        if execution_transport == "agent-sdk"
        else "test.native"
      ),
      route="test.in_process",
      lifecycle="active",
      capabilities={
        capability_id: (
          "user_selectable"
          if capability_id == "session.driver"
          else "internal"
        )
        for capability_id in CAPABILITY_IDS
      },
      supported_efforts=frozenset({bound_effort}),
      default_effort=bound_effort,
      features=frozenset({"tools", "streaming"}),
      reported_identities=frozenset({bound_model}),
    )
    model_registry = ProductModelRegistry(
      schema="product-model-registry/v1",
      revision="gateway-tests.1",
      models={model_key: registry_entry},
    )
    model_selection_policy = ProductModelSelectionPolicy(
      schema="product-model-selection/v1",
      revision="gateway-tests.1",
      capabilities={
        capability_id: CapabilitySelectionPolicy(
          capability_id=capability_id,
          default=CapabilityDefault(
            kind="model",
            model_key=model_key,
            effort=bound_effort,
          ),
          by_channel={},
          allowed_model_keys=frozenset({model_key}),
          allow_saved_preference=(capability_id == "session.driver"),
          allow_explicit_user=(capability_id == "session.driver"),
        )
        for capability_id in CAPABILITY_IDS
      },
    )
    service_handle = CredentialHandle(
      handle_id=f"service:gateway-tests:{provider_name}",
      provider=provider_name,
      principal="service",
      tenant_id="gateway-tests",
      actor_id=None,
    )
    service_material = MaterializedCredential(
      handle=service_handle,
      auth_config=bound_auth_config,
    )

    def _materialize_test_credential(
      handle: CredentialHandle,
    ) -> MaterializedCredential:
      if handle is not service_handle:
        raise RuntimeError("unknown gateway test credential handle")
      return service_material

    async def _resolve_test_credentials(
      _api_key: str,
      payload: Any,
    ) -> ResolverResult:
      raw_channel = (
        payload.context.get("channel")
        if isinstance(payload.context, dict)
        else None
      )
      channel = str(raw_channel or "web").strip().lower()
      return ResolverResult(
        user_id=str(payload.user_id or "alice"),
        channel=channel,
        auth_config=AuthConfig.from_dict(bound_auth_config),
        credential_principal="service",
        allow_service_for_interactive=True,
        risk_user_id=1,
        role="owner",
        model_entitled_capabilities=frozenset(CAPABILITY_IDS),
        model_entitled_keys=frozenset({model_key}),
      )

    async def _build_chat_runtime(session, request, channel, auth_manager):
      _ = channel, auth_manager
      state.session = session

      def _build_runner(
        event_log: EventLog,
        session_id: str,
        started_at: float,
      ):
        state.event_log = event_log
        kwargs = dict(runner_kwargs or {})
        capability_execution = request.capability_execution
        if capability_execution is None:
          raise RuntimeError(
            "gateway test runtime requires a capability execution"
          )

        if issubclass(runner_class, AgentSDKRunner):
          runner = runner_class(
            event_log=event_log,
            session_id=session_id,
            started_at=started_at,
            sdk_config=resolved_sdk_config,
            capability_execution=capability_execution,
            system_prompt="test",
            on_usage=None,
            on_tool_result=None,
            on_tool_timing=None,
            **kwargs,
          )
        else:
          if provider is None:
            raise ValueError("provider is required when runner_class is AgentRunner")

          session_auth_config = dict(request.bound_auth_config)
          kwargs.setdefault("user_id", session.user_id)
          kwargs.setdefault("billing_mode", str(session_auth_config.get("billing_mode") or "byok"))
          kwargs.setdefault("rate_table_version", str(session_auth_config.get("rate_table_version") or "unknown"))
          kwargs.setdefault("channel", session.channel)

          if dispatcher is None:
            dispatcher_obj = ToolDispatcher(
              mcp_client=_NullMcpClient(),
              local_tool_handlers={},
              event_log=event_log,
              session_id=session_id,
            )
          elif callable(dispatcher):
            dispatcher_obj = dispatcher(event_log, session_id, session)
          else:
            dispatcher_obj = dispatcher

          runner = runner_class(
            event_log=event_log,
            dispatcher=dispatcher_obj,
            session_id=session_id,
            started_at=started_at,
            capability_execution=capability_execution,
            get_tool_definitions=lambda: list(resolved_tool_definitions),
            per_turn_timeout=per_turn_timeout,
            stream_stall_timeout=stream_stall_timeout,
            tool_call_timeout=tool_call_timeout,
            **kwargs,
          )

        state.runner = runner
        return runner

      async def _on_disconnect() -> None:
        state.disconnect_hook_calls += 1
        if on_disconnect is not None:
          result = on_disconnect(state)
          if inspect.isawaitable(result):
            await result
        runner = state.runner
        if runner is None:
          return
        await runner.on_disconnect()

      runtime_kwargs: dict[str, Any] = {
        "system_prompt": "test",
        "build_runner": _build_runner,
        "capability_execution": request.capability_execution,
        "get_tool_definitions": lambda: list(resolved_tool_definitions),
      }
      if attach_disconnect_handler:
        runtime_kwargs["disconnect_handler"] = _on_disconnect

      runtime = ChatRuntime(
        **runtime_kwargs,
      )
      state.runtime = runtime
      return runtime

    app = create_gateway_app(
      GatewayServerConfig(
        jwt_secret="stream-lifecycle-test-secret-0123456789",
        valid_api_keys={"gateway-key"},
        default_provider=provider if execution_transport == "native" else None,
        tenant_id="gateway-tests",
        allow_service_credentials_for_interactive=True,
        credentials_resolver=_resolve_test_credentials,
        model_registry=model_registry,
        model_selection_policy=model_selection_policy,
        service_provider_handles={provider_name: service_handle},
        service_auth_config_resolver=_materialize_test_credential,
        capability_adapter_resolver=lambda selected_adapter: (
          provider
          if selected_adapter == adapter_id and provider is not None
          else AnthropicProvider()
        ),
        build_chat_runtime=_build_chat_runtime,
        transcript_dir=transcript_dir,
      )
    )
    app.state.test_state = state
    return app

  return _make_test_app
