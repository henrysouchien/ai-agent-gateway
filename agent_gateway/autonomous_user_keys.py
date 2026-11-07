"""User-key resolution over the autonomous child's existing session bearer.

The parent owns the Risk signer. The child can resolve/report only for the owner
of its live authenticated session; request bodies contain no identity selector.
"""
from __future__ import annotations

import asyncio
from dataclasses import asdict
from typing import Any

import httpx
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict

from hank_contracts.user_provider_keys import FailureContext, Provider, ResolveCredentialRefusal
from hank_model_transport.user_provider_keys import (
  NO_API_KEY_CONSUMER, IdentityUnresolved, ResolveContext,
  ResolvedUserKeys, UserKeyMissing, UserKeyPools, UserKeyResolverUnavailable,
  UserKeysSuspended, error_for_refusal, install_user_key_pools, installed_user_key_pools,
)
from .gateway_address import gateway_base_url
from .autonomous_launch_envelope import AUTONOMOUS_RUNTIME_SESSION_PURPOSE
from .session import AuthManager, GatewaySession

AUTONOMOUS_USER_KEYS_PATH = "/internal/autonomous/user-provider-keys"


class _Request(BaseModel):
  model_config = ConfigDict(extra="forbid", strict=True)
  provider: Provider
  failure: dict[str, Any] | None = None
  model: str | None = None


def build_autonomous_user_keys_router(*, auth: AuthManager) -> APIRouter:
  router = APIRouter()

  @router.post(AUTONOMOUS_USER_KEYS_PATH)
  async def resolve(request: Request, payload: _Request):
    session = auth.verify_token(auth.get_bearer_token(request.headers.get("authorization")))
    if session.purpose != AUTONOMOUS_RUNTIME_SESSION_PURPOSE:
      raise HTTPException(status_code=403, detail="autonomous_session_required")
    try:
      failure = FailureContext.from_wire(payload.failure) if payload.failure is not None else None
    except ValueError:
      raise HTTPException(status_code=422, detail="invalid_failure_context") from None
    context = ResolveContext(
      consumer_key_hash=NO_API_KEY_CONSUMER, channel="autonomous",
      session_id=session.session_id, model=payload.model,
    )
    try:
      if session.risk_user_id <= 0:
        raise IdentityUnresolved(payload.provider)
      source = installed_user_key_pools(payload.provider).source
      if failure is None:
        resolved = await asyncio.to_thread(
          source.resolve, risk_user_id=session.risk_user_id, provider=payload.provider, context=context,
        )
      else:
        resolved = await asyncio.to_thread(
          source.report, risk_user_id=session.risk_user_id, provider=payload.provider,
          context=context, failure=failure,
        )
    except UserKeysSuspended as exc:
      refusal = ResolveCredentialRefusal(
        "user_keys_suspended", provider=payload.provider, risk_user_id=session.risk_user_id,
        keys=exc.keys,
      )
      return JSONResponse(refusal.to_wire(), status_code=404)
    except UserKeyMissing as exc:
      refusal = ResolveCredentialRefusal(
        "user_key_missing", provider=payload.provider, risk_user_id=session.risk_user_id,
        message=exc.user_message,
      )
      return JSONResponse(refusal.to_wire(), status_code=404)
    except IdentityUnresolved:
      return JSONResponse(ResolveCredentialRefusal("identity_unresolved").to_wire(), status_code=400)
    except UserKeyResolverUnavailable:
      return JSONResponse(ResolveCredentialRefusal("resolver_transient").to_wire(), status_code=503)
    return JSONResponse(asdict(resolved), headers={"Cache-Control": "no-store"})

  return router


class AutonomousUserKeySource:
  """A run-scoped source, never the shared resolver HMAC credential."""

  def __init__(self, *, session: GatewaySession, base_url: str, post=None):
    self._risk_user_id = session.risk_user_id
    self._token = session.session_token
    self._url = base_url.rstrip("/") + AUTONOMOUS_USER_KEYS_PATH
    self._post = httpx.post if post is None else post

  def resolve(self, *, risk_user_id: int, provider: Provider, context: ResolveContext) -> ResolvedUserKeys:
    return self._resolve(risk_user_id, provider, context, None)

  def report(self, *, risk_user_id: int, provider: Provider, failure: FailureContext, context: ResolveContext) -> ResolvedUserKeys:
    return self._resolve(risk_user_id, provider, context, failure)

  def _resolve(self, risk_user_id, provider: Provider, context: ResolveContext, failure):
    if risk_user_id != self._risk_user_id:
      raise IdentityUnresolved(provider)
    if self._token is None:
      raise UserKeyResolverUnavailable(provider, "autonomous_session_token_missing")
    payload = _Request(provider=provider, failure=failure.to_wire() if failure is not None else None, model=context.model)
    try:
      response = self._post(
        self._url, json=payload.model_dump(),
        headers={"Authorization": "Bearer " + self._token},
      )
    except httpx.RequestError:
      raise UserKeyResolverUnavailable(provider, "autonomous_parent_unavailable") from None
    try:
      if response.status_code == 200:
        result = ResolvedUserKeys.from_payload(response.json())
        if result.risk_user_id != self._risk_user_id or result.provider != provider:
          raise ValueError("response identity mismatch")
        return result
      refusal = ResolveCredentialRefusal.from_wire(response.json())
    except ValueError:
      raise UserKeyResolverUnavailable(provider, "autonomous_parent_response_invalid") from None
    raise error_for_refusal(refusal, provider=provider, risk_user_id=self._risk_user_id)


def install_autonomous_user_key_pools(session: GatewaySession) -> None:
  install_user_key_pools(UserKeyPools(AutonomousUserKeySource(
    session=session, base_url=gateway_base_url(),
  )))
