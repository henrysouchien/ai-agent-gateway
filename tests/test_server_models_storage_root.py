from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import agent_gateway.server as server_module
from agent_gateway.sdk_runner import AgentSDKRunner
from agent_gateway.server_models import _call_build_chat_runtime


def _call(
  builder: Any,
  *,
  storage_root: Path | None,
) -> Any:
  return asyncio.run(
    _call_build_chat_runtime(
      builder,
      session="session",  # type: ignore[arg-type]
      request="request",  # type: ignore[arg-type]
      channel="web",
      auth_manager="auth",  # type: ignore[arg-type]
      storage_root=storage_root,
    )
  )


def test_chat_runtime_shim_forwards_storage_root_to_extension_builder(
  tmp_path: Path,
) -> None:
  async def builder(
    *,
    session: Any,
    request: Any,
    channel: Any,
    auth_manager: Any,
    storage_root: Path,
  ) -> tuple[Any, ...]:
    return session, request, channel, auth_manager, storage_root

  assert _call(builder, storage_root=tmp_path) == (
    "session",
    "request",
    "web",
    "auth",
    tmp_path,
  )


def test_chat_runtime_shim_degrades_for_positional_legacy_builder(
  tmp_path: Path,
) -> None:
  async def builder(
    session: Any,
    request: Any,
    channel: Any,
    auth_manager: Any,
    /,
  ) -> tuple[Any, ...]:
    return session, request, channel, auth_manager

  assert _call(builder, storage_root=tmp_path) == (
    "session",
    "request",
    "web",
    "auth",
  )


def test_chat_runtime_shim_degrades_for_keyword_only_legacy_builder(
  tmp_path: Path,
) -> None:
  async def builder(
    *,
    session: Any,
    request: Any,
    channel: Any,
    auth_manager: Any,
  ) -> tuple[Any, ...]:
    return session, request, channel, auth_manager

  assert _call(builder, storage_root=tmp_path) == (
    "session",
    "request",
    "web",
    "auth",
  )


def test_secondary_direct_shim_call_without_extension_is_unchanged() -> None:
  async def builder(
    *,
    session: Any,
    request: Any,
    channel: Any,
    auth_manager: Any,
  ) -> tuple[Any, ...]:
    return session, request, channel, auth_manager

  assert _call(builder, storage_root=None) == (
    "session",
    "request",
    "web",
    "auth",
  )


def test_gateway_request_builder_passes_app_state_storage_root(
  make_test_app: Any,
  monkeypatch: Any,
  tmp_path: Path,
) -> None:
  user_data_dir = tmp_path / "user-data"
  (user_data_dir / "gateway").mkdir(parents=True)
  monkeypatch.setenv("USER_DATA_DIR", str(user_data_dir))
  app = make_test_app(runner_class=AgentSDKRunner)
  captured: dict[str, Any] = {}

  async def fake_call(
    _builder: Any,
    **kwargs: Any,
  ) -> Any:
    captured.update(kwargs)
    return SimpleNamespace(purpose=None)

  monkeypatch.setattr(
    server_module,
    "_call_build_chat_runtime",
    fake_call,
  )
  session = SimpleNamespace(purpose="chat")

  runtime = asyncio.run(
    app.state.gateway_build_chat_runtime(
      session=session,
      request=object(),
      channel="web",
      auth_manager=app.state.auth,
    )
  )

  assert runtime.purpose == "chat"
  assert (
    captured["storage_root"]
    == app.state.autonomous_storage_root
  )
