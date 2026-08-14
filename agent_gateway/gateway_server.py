from __future__ import annotations

import argparse
import ctypes
import resource
import sys
from typing import Sequence

import uvicorn

from .claim_signing_authority import (
  GatewayClaimSigningAuthority,
  install_gateway_claim_signing_authority,
)


_GATEWAY_APP = "main:app"
_TCP_HOST = "127.0.0.1"
_TCP_PORT = 8001
_PR_GET_DUMPABLE = 3
_PR_SET_DUMPABLE = 4


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
  claim_signing_key_fd: int,
  workers: int = 1,
  timeout_keep_alive: int = 120,
  app: str = _GATEWAY_APP,
  host: str = _TCP_HOST,
  port: int = _TCP_PORT,
  ssl_keyfile: str | None = None,
  ssl_certfile: str | None = None,
) -> None:
  if workers != 1:
    raise ValueError(
      "the gateway launcher requires exactly one worker"
    )
  if (
    isinstance(timeout_keep_alive, bool)
    or not isinstance(timeout_keep_alive, int)
    or timeout_keep_alive <= 0
  ):
    raise ValueError(
      "timeout_keep_alive must be a positive integer"
    )
  if (ssl_keyfile is None) != (ssl_certfile is None):
    raise ValueError(
      "ssl_keyfile and ssl_certfile must be provided together"
    )
  harden_claim_signing_process_boundary()
  authority = GatewayClaimSigningAuthority.from_one_shot_fd(
    claim_signing_key_fd
  )
  install_gateway_claim_signing_authority(authority)
  config_kwargs: dict[str, object] = {
    "host": host,
    "port": port,
    "workers": 1,
    "timeout_keep_alive": timeout_keep_alive,
  }
  if ssl_keyfile is not None:
    config_kwargs["ssl_keyfile"] = ssl_keyfile
    config_kwargs["ssl_certfile"] = ssl_certfile
  config = uvicorn.Config(app, **config_kwargs)
  uvicorn.Server(config).run()


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
