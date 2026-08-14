import asyncio
import sys
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[3]
PKG_DIR = ROOT / "packages" / "agent-gateway"
if str(PKG_DIR) not in sys.path:
  sys.path.insert(0, str(PKG_DIR))

from agent_gateway import (  # noqa: E402
  AgentRunner,
  EventLog,
  ModelInfo,
  ModelProvider,
  ToolDispatcher,
)
import agent_gateway.runner as gateway_runner  # noqa: E402
from agent_gateway.auth import ProviderCredentialFailure  # noqa: E402
from agent_gateway.runner_auth import call_credential_refresher, merge_refreshed_auth_config  # noqa: E402
from tests.capability_execution_test_support import (  # noqa: E402
  stub_bound_capability_execution,
)


class _NullMcpClient:
  def is_mcp_tool(self, _name: str) -> bool:
    return False

  async def call_tool(self, name: str, _tool_input: dict[str, Any]):
    return None, {"code": "unknown_tool", "message": f"Unknown tool: {name}"}

  def get_tool_definitions(self) -> list[dict[str, Any]]:
    return []


class _StubProvider(ModelProvider):
  name = "stub"

  def has_active_credential(self, config: dict[str, Any]) -> bool:
    return bool(config.get("api_key"))

  def get_model_info(self, model: str) -> ModelInfo:
    return ModelInfo(
      id=model,
      provider=self.name,
      max_output_tokens=4096,
      supports_thinking=True,
    )


class _Logger:
  def __init__(self) -> None:
    self.warnings: list[tuple[Any, ...]] = []

  def warning(self, *args: Any) -> None:
    self.warnings.append(args)


def _run(coro):
  return asyncio.run(coro)


def _failure(*, retryable: bool = True) -> ProviderCredentialFailure:
  return ProviderCredentialFailure(
    provider="stub",
    kind="auth",
    status_code=401,
    retryable_with_new_credentials=retryable,
  )


def _make_dispatcher(event_log: EventLog | None = None) -> ToolDispatcher:
  return ToolDispatcher(
    mcp_client=_NullMcpClient(),
    local_tool_handlers={},
    event_log=event_log or EventLog(),
    session_id="sess-auth",
  )


def _make_runner(*, on_credential_failure=None) -> AgentRunner:
  event_log = EventLog()
  provider = _StubProvider()
  return AgentRunner(
    event_log=event_log,
    dispatcher=_make_dispatcher(event_log),
    session_id="sess-auth",
    capability_execution=stub_bound_capability_execution(
      provider=provider,
      model="stub-model",
      effort="none",
      auth_config={"api_key": "old", "max_tokens": 512},
    ),
    on_credential_failure=on_credential_failure,
    user_id="alice",
    billing_mode="byok",
    rate_table_version="unknown",
  )


def test_runner_auth_wrappers_resolve_parent_module_helpers(monkeypatch: Any) -> None:
  runner = _make_runner()
  original_execution = runner.capability_execution
  config = dict(original_execution.auth_config)
  config["api_key"] = "request-old"

  monkeypatch.setattr(
    gateway_runner,
    "_merge_refreshed_auth_config",
    lambda original, refreshed: {
      **original,
      "api_key": f"{original['api_key']}->{refreshed['api_key']}",
    },
  )

  AgentRunner._apply_refreshed_auth_config(runner, config, {"api_key": "new"})

  assert config["api_key"] == "old->new"
  assert runner._auth_config["api_key"] == "old->new"
  assert runner.capability_execution is not original_execution
  assert runner.capability_execution.bind is original_execution.bind
  assert runner.capability_execution.provider is original_execution.provider
  assert runner._secret_boundary.sanitize(
    ["old", "old->new"],
    sink="test",
  ) == ["<redacted-secret>", "<redacted-secret>"]


def test_merge_refreshed_auth_config_preserves_runtime_controls() -> None:
  merged = merge_refreshed_auth_config(
    {
      "api_key": "old",
      "auth_mode": "oauth",
      "auth_token": "old-token",
      "provider": "stub",
      "max_tokens": 4096,
      "billing_mode": "metered",
      "rate_table_version": "2026-04-08",
    },
    {
      "api_key": "new",
      "auth_mode": "oauth",
      "auth_token": "new-token",
      "billing_mode": "byok",
      "rate_table_version": "unknown",
    },
  )

  assert merged == {
    "api_key": "new",
    "auth_mode": "oauth",
    "auth_token": "new-token",
    "provider": "stub",
    "max_tokens": 4096,
    "billing_mode": "metered",
    "rate_table_version": "2026-04-08",
  }


def test_merge_refreshed_auth_config_rejects_selection_material() -> None:
  config = {
    "api_key": "old",
    "auth_mode": "api",
    "provider": "stub",
    "max_tokens": 4096,
  }

  with pytest.raises(ValueError, match="must not contain model selection"):
    merge_refreshed_auth_config(config, {"model": "other-model"})
  with pytest.raises(ValueError, match="must not contain model selection"):
    merge_refreshed_auth_config(config, {"thinking": False})
  with pytest.raises(ValueError, match="must not contain model selection"):
    merge_refreshed_auth_config(config, {"execution_transport": "native"})


def test_credential_refresher_helper_supports_sync_and_async_callbacks() -> None:
  metrics: list[tuple[str, int]] = []
  logger = _Logger()

  sync_result = _run(
    call_credential_refresher(
      lambda failure: {"api_key": f"new-{failure.kind}"},
      _failure(),
      emit_metric=lambda name, value: metrics.append((name, value)),
      log_session_id="sess-auth",
      logger=logger,
    )
  )

  async def _async_refresher(_failure: ProviderCredentialFailure) -> dict[str, Any]:
    return {"auth_token": "new-token"}

  async_result = _run(
    call_credential_refresher(
      _async_refresher,
      _failure(),
      emit_metric=lambda name, value: metrics.append((name, value)),
      log_session_id="sess-auth",
      logger=logger,
    )
  )

  assert sync_result == {"api_key": "new-auth"}
  assert async_result == {"auth_token": "new-token"}
  assert metrics == []
  assert logger.warnings == []


def test_credential_refresher_helper_records_unavailable_and_failed_metrics() -> None:
  metrics: list[tuple[str, int]] = []
  logger = _Logger()

  unavailable = _run(
    call_credential_refresher(
      lambda _failure: {},
      _failure(),
      emit_metric=lambda name, value: metrics.append((name, value)),
      log_session_id="sess-auth",
      logger=logger,
    )
  )

  def _failing_refresher(_failure: ProviderCredentialFailure) -> dict[str, Any]:
    raise RuntimeError("resolver offline")

  failed = _run(
    call_credential_refresher(
      _failing_refresher,
      _failure(),
      emit_metric=lambda name, value: metrics.append((name, value)),
      log_session_id="sess-auth",
      logger=logger,
    )
  )

  skipped = _run(
    call_credential_refresher(
      lambda _failure: {"api_key": "new"},
      _failure(retryable=False),
      emit_metric=lambda name, value: metrics.append((name, value)),
      log_session_id="sess-auth",
      logger=logger,
    )
  )

  assert unavailable is None
  assert failed is None
  assert skipped is None
  assert metrics == [
    ("gateway.credential_refresh_unavailable", 1),
    ("gateway.credential_refresh_failed", 1),
  ]
  assert logger.warnings
  assert "credential refresh failed" in logger.warnings[0][0]


def test_runner_credential_refresher_delegate_records_metrics() -> None:
  metrics: list[tuple[str, int]] = []
  runner = _make_runner(on_credential_failure=lambda _failure: {"api_key": "new"})
  runner._on_metric = lambda name, value: metrics.append((name, value))

  assert _run(runner._call_credential_refresher(_failure())) == {"api_key": "new"}
  assert metrics == []

  runner.set_credential_refresher(lambda _failure: {})

  assert _run(runner._call_credential_refresher(_failure())) is None
  assert metrics == [("gateway.credential_refresh_unavailable", 1)]
