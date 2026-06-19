import asyncio
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
PKG_DIR = ROOT / "packages" / "agent-gateway"
if str(PKG_DIR) not in sys.path:
  sys.path.insert(0, str(PKG_DIR))

from agent_gateway import AgentRunner, EventLog, ToolDispatcher  # noqa: E402
import agent_gateway.runner as gateway_runner  # noqa: E402
from agent_gateway.auth import ProviderCredentialFailure  # noqa: E402
from agent_gateway.runner_auth import call_credential_refresher, merge_refreshed_auth_config  # noqa: E402


class _NullMcpClient:
  def is_mcp_tool(self, _name: str) -> bool:
    return False

  async def call_tool(self, name: str, _tool_input: dict[str, Any]):
    return None, {"code": "unknown_tool", "message": f"Unknown tool: {name}"}

  def get_tool_definitions(self) -> list[dict[str, Any]]:
    return []


class _StubProvider:
  name = "stub"


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
  return AgentRunner(
    event_log=event_log,
    dispatcher=_make_dispatcher(event_log),
    session_id="sess-auth",
    provider=_StubProvider(),
    auth_config={"api_key": "old", "model": "stub-model"},
    on_credential_failure=on_credential_failure,
    user_id="alice",
    billing_mode="byok",
    rate_table_version="unknown",
  )


def test_runner_auth_wrappers_resolve_parent_module_helpers(monkeypatch: Any) -> None:
  runner = object.__new__(AgentRunner)
  runner._auth_config = {"api_key": "old"}
  config = {"api_key": "request-old"}

  monkeypatch.setattr(
    gateway_runner,
    "_merge_refreshed_auth_config",
    lambda original, refreshed: {"api_key": f"{original['api_key']}->{refreshed['api_key']}"},
  )

  AgentRunner._apply_refreshed_auth_config(runner, config, {"api_key": "new"})

  assert config == {"api_key": "request-old->new"}
  assert runner._auth_config == {"api_key": "request-old->new"}


def test_merge_refreshed_auth_config_preserves_runtime_controls() -> None:
  merged = merge_refreshed_auth_config(
    {
      "api_key": "old",
      "auth_mode": "oauth",
      "auth_token": "old-token",
      "model": "chosen-model",
      "max_tokens": 4096,
      "thinking": False,
      "billing_mode": "metered",
      "rate_table_version": "2026-04-08",
    },
    {
      "api_key": "new",
      "auth_mode": "api",
      "auth_token": "",
      "model": "ignored-model",
      "max_tokens": 2000,
      "thinking": True,
      "billing_mode": "byok",
      "rate_table_version": "unknown",
    },
  )

  assert merged == {
    "api_key": "new",
    "auth_mode": "api",
    "auth_token": "",
    "model": "chosen-model",
    "max_tokens": 4096,
    "thinking": False,
    "billing_mode": "metered",
    "rate_table_version": "2026-04-08",
  }


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
