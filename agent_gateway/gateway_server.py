# TCP launch boundary for both supported launches: on the privileged host
# privileged_claim_launcher hands off a claim-signing fd, and the local
# immutable child (scripts/local_gateway_python_child.py, target
# research-gateway-uvicorn) calls run_gateway_server directly. This boundary
# delegates to uvicorn.Config/Server.run with exactly one worker. Uvicorn 0.53
# app-load/bind/lifespan failures exit 3 and are logged here; systemd
# Restart=always still restarts them. A startup that raises
# GatewaySchemaIncompatibleError exits 4 (failure_class=schema), which
# gateway.service names in RestartPreventExitStatus: a store this code cannot
# serve does not change on restart.
# Owners: packages/agent-gateway/README.md;
# docs/reference/local-gateway-immutable-runtime.md.

from __future__ import annotations

import argparse
import ctypes
import logging
import resource
import sys
from typing import Any, Sequence

import uvicorn

from .claim_signing_authority import (
  GatewayClaimSigningAuthority,
  install_gateway_claim_signing_authority,
)
from .startup_failures import (
  SCHEMA_INCOMPATIBLE_EXIT_CODE,
  GatewaySchemaIncompatibleError,
)


_GATEWAY_APP = "main:app"
_TCP_HOST = "127.0.0.1"
_TCP_PORT = 8001
_PR_GET_DUMPABLE = 3
_PR_SET_DUMPABLE = 4
class _LifespanFailureRecorder:
  """ASGI wrapper that keeps the exception the app's lifespan raised.

  Uvicorn logs a failed startup and drops the exception, so the launch
  boundary records it here to classify the failure.
  """

  def __init__(self, app: Any) -> None:
    self.app = app
    self.failure: BaseException | None = None

  async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
    if scope["type"] != "lifespan":
      await self.app(scope, receive, send)
      return
    try:
      await self.app(scope, receive, send)
    except BaseException as exc:
      self.failure = exc
      raise


def harden_claim_signing_process_boundary() -> None:
  """Close same-UID memory/core-dump access before adopting the root key."""

  resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
  if resource.getrlimit(resource.RLIMIT_CORE) != (0, 0):
    raise RuntimeError(
      "gateway could not disable core dumps"
    )
  if sys.platform != "linux":
    return
  libc = ctypes.CDLL(None, use_errno=True)
  if libc.prctl(_PR_SET_DUMPABLE, 0, 0, 0, 0) != 0:
    error = ctypes.get_errno()
    raise OSError(
      error,
      "gateway could not disable dumpable state",
    )
  if libc.prctl(_PR_GET_DUMPABLE, 0, 0, 0, 0) != 0:
    raise RuntimeError(
      "gateway remained dumpable before key adoption"
    )


def run_gateway_server(
  *,
  claim_signing_key_fd: int | None,
  workers: int = 1,
  timeout_keep_alive: int = 120,
  timeout_graceful_shutdown: int = 30,
  app: str = _GATEWAY_APP,
  host: str = _TCP_HOST,
  port: int = _TCP_PORT,
  ssl_keyfile: str | None = None,
  ssl_certfile: str | None = None,
) -> None:
  # Sessions, bearer verification and active turns live in this process's
  # SessionStore memory, so a second worker would answer another worker's
  # bearer as unknown and admit a second turn on the same session.
  if workers != 1:
    raise ValueError(
      "the gateway launcher requires exactly one worker (session and turn "
      "state is in-process); launch with workers=1"
    )
  if (
    isinstance(timeout_keep_alive, bool)
    or not isinstance(timeout_keep_alive, int)
    or timeout_keep_alive <= 0
  ):
    raise ValueError(
      "timeout_keep_alive must be a positive integer"
    )
  if (
    isinstance(timeout_graceful_shutdown, bool)
    or not isinstance(timeout_graceful_shutdown, int)
    or timeout_graceful_shutdown <= 0
  ):
    raise ValueError(
      "timeout_graceful_shutdown must be a positive integer"
    )
  if (ssl_keyfile is None) != (ssl_certfile is None):
    raise ValueError(
      "ssl_keyfile and ssl_certfile must be provided together"
    )
  # A missing fd is the degraded local boot the launcher already reports
  # ("claim-signing authority unavailable ..."): the gateway serves, and
  # autonomous dispatch stays refused for want of installed authority. The
  # privileged path's --claim-signing-key-fd is required, so it never lands here.
  if claim_signing_key_fd is not None:
    harden_claim_signing_process_boundary()
    authority = GatewayClaimSigningAuthority.from_one_shot_fd(
      claim_signing_key_fd
    )
    install_gateway_claim_signing_authority(authority)
  if ssl_keyfile is None:
    config = uvicorn.Config(
      app,
      host=host,
      port=port,
      workers=1,
      timeout_keep_alive=timeout_keep_alive,
      timeout_graceful_shutdown=timeout_graceful_shutdown,
    )
  else:
    config = uvicorn.Config(
      app,
      host=host,
      port=port,
      workers=1,
      timeout_keep_alive=timeout_keep_alive,
      timeout_graceful_shutdown=timeout_graceful_shutdown,
      ssl_keyfile=ssl_keyfile,
      ssl_certfile=ssl_certfile,
    )
  recorder: list[_LifespanFailureRecorder] = []
  load_app = config.load

  def load_recording_lifespan() -> None:
    load_app()
    recorder.append(_LifespanFailureRecorder(config.loaded_app))
    config.loaded_app = recorder[-1]

  setattr(config, "load", load_recording_lifespan)
  server = uvicorn.Server(config)
  try:
    server.run()
    if not server.started:
      raise SystemExit(3)
  except (SystemExit, Exception) as exc:
    exit_code = 1
    if isinstance(exc, SystemExit):
      exit_code = exc.code if isinstance(exc.code, int) else int(exc.code is not None)
    if not exit_code:
      if server.started:
        raise
      exit_code = 3
    if server.started:
      failure_class = "runtime"
    elif not config.loaded:
      failure_class = "app_load"
    elif (
      server.lifespan.startup_failed
      or (server.lifespan.error_occurred and config.lifespan == "on")
    ):
      failure_class = "lifespan"
      if recorder and isinstance(recorder[-1].failure, GatewaySchemaIncompatibleError):
        failure_class = "schema"
        exit_code = SCHEMA_INCOMPATIBLE_EXIT_CODE
    else:
      failure_class = "bind"
    logging.getLogger(__name__).error(
      "event=gateway_launch_failed failure_class=%s exit_code=%s",
      failure_class,
      exit_code,
    )
    if failure_class == "schema" or (isinstance(exc, SystemExit) and not exc.code):
      raise SystemExit(exit_code) from exc
    raise


def main(argv: Sequence[str] | None = None) -> int:
  parser = argparse.ArgumentParser(
    description=(
      "Run the gateway on its ordinary TCP listener."
    )
  )
  parser.add_argument(
    "--claim-signing-key-fd",
    type=int,
    required=True,
  )
  parser.add_argument(
    "--workers",
    type=int,
    choices=(1,),
    default=1,
  )
  parser.add_argument(
    "--timeout-keep-alive",
    type=int,
    default=120,
  )
  parser.add_argument(
    "--timeout-graceful-shutdown",
    type=int,
    default=30,
  )
  parser.add_argument("--app", default=_GATEWAY_APP)
  parser.add_argument("--host", default=_TCP_HOST)
  parser.add_argument("--port", type=int, default=_TCP_PORT)
  parser.add_argument("--ssl-keyfile", default=None)
  parser.add_argument("--ssl-certfile", default=None)
  args = parser.parse_args(argv)
  run_gateway_server(
    claim_signing_key_fd=args.claim_signing_key_fd,
    workers=args.workers,
    timeout_keep_alive=args.timeout_keep_alive,
    timeout_graceful_shutdown=args.timeout_graceful_shutdown,
    app=args.app,
    host=args.host,
    port=args.port,
    ssl_keyfile=args.ssl_keyfile,
    ssl_certfile=args.ssl_certfile,
  )
  return 0


if __name__ == "__main__":
  raise SystemExit(main())


__all__ = [
  "harden_claim_signing_process_boundary",
  "main",
  "run_gateway_server",
]
