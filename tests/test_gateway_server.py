from __future__ import annotations

import os
import resource

import pytest

from agent_gateway import gateway_server


def _claim_signing_key_fd() -> int:
  read_fd, write_fd = os.pipe()
  os.write(
    write_fd,
    b"gateway-server-test-claim-key-at-least-32-bytes",
  )
  os.close(write_fd)
  return read_fd


def test_run_gateway_server_uses_ordinary_tcp_server(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  seen: dict[str, object] = {}

  class _Config:
    def __init__(self, app: str, **kwargs: object) -> None:
      seen["app"] = app
      seen["config"] = dict(kwargs)

  class _Server:
    def __init__(self, config: _Config) -> None:
      seen["server_config"] = config

    def run(self) -> None:
      seen["ran"] = True

  monkeypatch.setattr(gateway_server.uvicorn, "Config", _Config)
  monkeypatch.setattr(gateway_server.uvicorn, "Server", _Server)
  installed: list[object] = []
  monkeypatch.setattr(
    gateway_server,
    "install_gateway_claim_signing_authority",
    installed.append,
  )

  gateway_server.run_gateway_server(
    claim_signing_key_fd=_claim_signing_key_fd(),
    workers=1,
    timeout_keep_alive=37,
    timeout_graceful_shutdown=29,
  )

  assert seen["app"] == "main:app"
  assert seen["config"] == {
    "host": "127.0.0.1",
    "port": 8001,
    "workers": 1,
    "timeout_keep_alive": 37,
    "timeout_graceful_shutdown": 29,
  }
  assert seen["ran"] is True
  assert seen["server_config"] is not None
  assert len(installed) == 1


def test_run_gateway_server_supports_tls_serving_shape(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  seen: dict[str, object] = {}

  class _Config:
    def __init__(self, app: str, **kwargs: object) -> None:
      seen["app"] = app
      seen["config"] = dict(kwargs)

  class _Server:
    def __init__(self, config: _Config) -> None:
      seen["server_config"] = config

    def run(self) -> None:
      seen["ran"] = True

  monkeypatch.setattr(gateway_server.uvicorn, "Config", _Config)
  monkeypatch.setattr(gateway_server.uvicorn, "Server", _Server)
  installed: list[object] = []
  monkeypatch.setattr(
    gateway_server,
    "install_gateway_claim_signing_authority",
    installed.append,
  )

  gateway_server.run_gateway_server(
    claim_signing_key_fd=_claim_signing_key_fd(),
    app="api.main:app",
    host="0.0.0.0",
    port=8000,
    ssl_keyfile="/tmp/localhost.key",
    ssl_certfile="/tmp/localhost.crt",
  )

  assert seen["app"] == "api.main:app"
  assert seen["config"] == {
    "host": "0.0.0.0",
    "port": 8000,
    "workers": 1,
    "timeout_keep_alive": 120,
    "timeout_graceful_shutdown": 30,
    "ssl_keyfile": "/tmp/localhost.key",
    "ssl_certfile": "/tmp/localhost.crt",
  }
  assert seen["ran"] is True
  assert len(installed) == 1


@pytest.mark.parametrize(
  ("ssl_keyfile", "ssl_certfile"),
  [("/tmp/localhost.key", None), (None, "/tmp/localhost.crt")],
)
def test_run_gateway_server_rejects_partial_tls_pair(
  ssl_keyfile: str | None,
  ssl_certfile: str | None,
) -> None:
  with pytest.raises(ValueError):
    gateway_server.run_gateway_server(
      claim_signing_key_fd=-1,
      ssl_keyfile=ssl_keyfile,
      ssl_certfile=ssl_certfile,
    )


@pytest.mark.parametrize(
  ("workers", "timeout_keep_alive", "timeout_graceful_shutdown"),
  [
    (2, 120, 30),
    (1, 0, 30),
    (1, -1, 30),
    (1, True, 30),
    (1, 120, 0),
    (1, 120, -1),
    (1, 120, True),
  ],
)
def test_run_gateway_server_rejects_unsafe_process_shape(
  workers: int,
  timeout_keep_alive: int,
  timeout_graceful_shutdown: int,
) -> None:
  with pytest.raises(ValueError):
    gateway_server.run_gateway_server(
      claim_signing_key_fd=-1,
      workers=workers,
      timeout_keep_alive=timeout_keep_alive,
      timeout_graceful_shutdown=timeout_graceful_shutdown,
    )


def test_claim_boundary_disables_core_dumps_before_fd_adoption() -> None:
  gateway_server.harden_claim_signing_process_boundary()
  assert resource.getrlimit(resource.RLIMIT_CORE) == (0, 0)


def test_cli_forwards_graceful_shutdown_bound(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  seen: dict[str, object] = {}
  monkeypatch.setattr(
    gateway_server,
    "run_gateway_server",
    lambda **kwargs: seen.update(kwargs),
  )

  assert gateway_server.main([
    "--claim-signing-key-fd",
    "7",
    "--timeout-graceful-shutdown",
    "30",
  ]) == 0
  assert seen["claim_signing_key_fd"] == 7
  assert seen["timeout_graceful_shutdown"] == 30
