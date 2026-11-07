"""Signed v2 client for Risk's user-owned provider keys (design §§3, 4, 6).

Serialize once: the exact bytes bound by the HMAC are posted as ``content``.
Risk owns key material and permanent health; this client never tries another store.
"""

from __future__ import annotations

import hashlib
from importlib.resources import files
import hmac
import json
import logging
import time
from typing import Any, Callable, Literal, Mapping
import uuid

import httpx
from hank_contracts.user_provider_keys import (
  RESOLVE_CREDENTIAL_PATH,
  RESOLVE_REFUSAL_CODES,
  USER_PROVIDER_KEYS_PATH,
  AddKeyRequest,
  EnrolmentOk,
  EnrolmentRefusal,
  FailureContext,
  KeyStatusRequest,
  ListKeysRequest,
  Provider,
  ResolveCredentialOk,
  ResolveCredentialRefusal,
  ResolveCredentialRequest,
  StatusOperation,
)
from hank_model_transport.user_provider_keys import (
  IdentityUnresolved,
  ResolveContext,
  ResolvedUserKeys,
  UserKeyResolverUnavailable,
  error_for_refusal,
)


log = logging.getLogger(__name__)
SyncPost = Callable[[str, bytes, dict[str, str], float], tuple[int, bytes]]
Operation = Literal["add", "list", "disable", "enable", "remove"]


def _declared_resolve_refusal_status() -> dict[str, int]:
  """HTTP status for each resolve refusal code the v2 contract declares.

  Codes come from ``RESOLVE_REFUSAL_CODES``. Statuses come from the refusal
  schema description that declares them. A code the contract adds without a
  declared status is absent here; lookup then refuses the reply instead of
  raising ``KeyError``.
  """

  raw = json.loads(
    files("hank_contracts").joinpath(
      "contracts/user-provider-keys-v2/resolve-credential.v2.schema.json",
    ).read_text(encoding="utf-8")
  )
  if not isinstance(raw, dict):
    return {}
  defs = raw.get("$defs")
  refusal = defs.get("Refusal") if isinstance(defs, dict) else None
  description = refusal.get("description") if isinstance(refusal, dict) else None
  if not isinstance(description, str) or not description.startswith("HTTP "):
    return {}
  found: dict[str, int] = {}
  for clause in description.removeprefix("HTTP ").rstrip(".").split(";"):
    head, _, names = clause.strip().partition(" ")
    if not head.isdigit() or not names:
      continue
    status = int(head)
    for name in names.split("|"):
      code = name.strip()
      if code:
        found[code] = status
  return {code: found[code] for code in RESOLVE_REFUSAL_CODES if code in found}


_RESOLVE_REFUSAL_STATUS = _declared_resolve_refusal_status()


def encode_body(document: Mapping[str, Any]) -> bytes:
  return json.dumps(document, separators=(",", ":"), ensure_ascii=False, sort_keys=True).encode("utf-8")


def v2_signature_payload(
  *, timestamp: str, risk_user_id: int, provider: str, consumer_key_hash: str, body: bytes,
) -> str:
  return f"v2\n{timestamp}\n{risk_user_id}\n{provider}\n{consumer_key_hash}\n{hashlib.sha256(body).hexdigest()}"


def signed_v2_headers(
  *, hmac_key: str, risk_user_id: int, provider: str, consumer_key_hash: str,
  body: bytes, timestamp: str | None = None,
) -> dict[str, str]:
  if timestamp is None:
    timestamp = str(int(time.time()))
  payload = v2_signature_payload(
    timestamp=timestamp, risk_user_id=risk_user_id, provider=provider,
    consumer_key_hash=consumer_key_hash, body=body,
  )
  return {
    "Content-Type": "application/json",
    "X-Resolver-Timestamp": timestamp,
    "X-Resolver-Signature": hmac.new(hmac_key.encode("utf-8"), payload.encode("utf-8"), hashlib.sha256).hexdigest(),
  }


def internal_route_url(resolve_url: str, operation: Operation) -> str:
  if not resolve_url.endswith(RESOLVE_CREDENTIAL_PATH):
    raise ValueError("resolve_url must end with RESOLVE_CREDENTIAL_PATH")
  return resolve_url[:-len(RESOLVE_CREDENTIAL_PATH)] + USER_PROVIDER_KEYS_PATH + "/" + operation


def _http_post(url: str, body: bytes, headers: dict[str, str], timeout: float) -> tuple[int, bytes]:
  with httpx.Client(timeout=timeout, trust_env=False) as client:
    response = client.post(url, content=body, headers=headers)
    return response.status_code, response.content


class EnrolmentRefused(RuntimeError):
  """A parsed enrolment refusal, rendered without the server's free-form message."""

  def __init__(self, refusal: EnrolmentRefusal) -> None:
    self.code = refusal.error
    self.fingerprint = refusal.fingerprint
    self.user_message = {
      "key_invalid": "the provider rejected this API key",
      "workspace_required": "this key needs a workspace id",
      "key_already_enrolled": "this key is already connected",
      "key_not_found": "no connected key has that fingerprint",
      "identity_unresolved": "this account is not linked to a Hank user, so no API key can be resolved for it",
    }.get(refusal.error, "the API key service is unavailable; try again shortly")
    super().__init__(self.user_message)


class RiskUserKeyClient:
  def __init__(
    self, *, resolve_url: str, hmac_key: str, timeout_seconds: float = 3.0,
    post: SyncPost | None = None,
  ) -> None:
    if not resolve_url:
      raise ValueError("resolve_url must not be empty")
    if not hmac_key:
      raise ValueError("hmac_key must not be empty")
    if not resolve_url.endswith(RESOLVE_CREDENTIAL_PATH):
      raise ValueError("resolve_url must end with RESOLVE_CREDENTIAL_PATH")
    self._resolve_url = resolve_url
    self._hmac_key = hmac_key
    self._timeout_seconds = timeout_seconds
    self._post = _http_post if post is None else post

  def _send(
    self, *, url: str, document: Mapping[str, Any], risk_user_id: int,
    provider: Provider, consumer_key_hash: str,
  ) -> tuple[int, bytes]:
    body = encode_body(document)
    headers = signed_v2_headers(
      hmac_key=self._hmac_key, risk_user_id=risk_user_id, provider=provider,
      consumer_key_hash=consumer_key_hash, body=body,
    )
    try:
      return self._post(url, body, headers, self._timeout_seconds)
    except httpx.TimeoutException:
      log.warning("User key service code=timeout")
      raise UserKeyResolverUnavailable(provider, "timeout") from None
    except httpx.RequestError:
      log.warning("User key service code=request")
      raise UserKeyResolverUnavailable(provider, "request") from None

  def resolve(
    self, *, risk_user_id: int, provider: Provider, context: ResolveContext,
  ) -> ResolvedUserKeys:
    return self._resolve(risk_user_id=risk_user_id, provider=provider, context=context, failure=None)

  def report(
    self, *, risk_user_id: int, provider: Provider, failure: FailureContext, context: ResolveContext,
  ) -> ResolvedUserKeys:
    return self._resolve(risk_user_id=risk_user_id, provider=provider, context=context, failure=failure)

  def _resolve(
    self, *, risk_user_id: int, provider: Provider, context: ResolveContext,
    failure: FailureContext | None,
  ) -> ResolvedUserKeys:
    request = ResolveCredentialRequest(
      risk_user_id=risk_user_id, provider=provider, consumer_key_hash=context.consumer_key_hash,
      request_id=context.request_id, channel=context.channel, session_id=context.session_id,
      model=context.model, refresh=failure is not None, failure=failure,
    )
    status, body = self._send(
      url=self._resolve_url, document=request.to_wire(), risk_user_id=risk_user_id,
      provider=provider, consumer_key_hash=context.consumer_key_hash,
    )
    try:
      document = json.loads(body)
      if status == 200:
        ok = ResolveCredentialOk.from_wire(document)
        if ok.provider != provider or ok.risk_user_id != risk_user_id:
          raise ValueError("response identity mismatch")
        log.info("User key resolution fingerprints=%s", ok.fingerprints)
        return ResolvedUserKeys.from_contract(ok)
      refusal = ResolveCredentialRefusal.from_wire(document)
      expected = _RESOLVE_REFUSAL_STATUS.get(refusal.error)
      if (
        expected is None
        or expected != status
        or (refusal.provider is not None and refusal.provider != provider)
        or (refusal.risk_user_id is not None and refusal.risk_user_id != risk_user_id)
      ):
        raise ValueError("refusal status or identity mismatch")
    except (ValueError, KeyError):
      log.warning("User key service code=resolver_response_invalid")
      raise UserKeyResolverUnavailable(provider, "resolver_response_invalid") from None
    log.info("User key resolution code=%s", refusal.error)
    raise error_for_refusal(refusal, provider=provider, risk_user_id=risk_user_id)

  def add_key(
    self, *, risk_user_id: int, provider: Provider, api_key: str, label: str,
    workspace_id: str | None = None,
  ) -> EnrolmentOk:
    request = AddKeyRequest(
      risk_user_id=risk_user_id, provider=provider, request_id=str(uuid.uuid4()),
      api_key=api_key, label=label, workspace_id=workspace_id,
    )
    return self._enrolment(request, "add")

  def list_keys(self, *, risk_user_id: int, provider: Provider) -> EnrolmentOk:
    return self._enrolment(ListKeysRequest(
      risk_user_id=risk_user_id, provider=provider, request_id=str(uuid.uuid4()),
    ), "list")

  def set_key_status(
    self, *, risk_user_id: int, provider: Provider, fingerprint: str, operation: StatusOperation,
  ) -> EnrolmentOk:
    return self._enrolment(KeyStatusRequest(
      risk_user_id=risk_user_id, provider=provider, request_id=str(uuid.uuid4()), fingerprint=fingerprint,
    ), operation)

  def _enrolment(
    self, request: AddKeyRequest | ListKeysRequest | KeyStatusRequest, operation: Operation,
  ) -> EnrolmentOk:
    status, body = self._send(
      url=internal_route_url(self._resolve_url, operation), document=request.to_wire(),
      risk_user_id=request.risk_user_id, provider=request.provider, consumer_key_hash="",
    )
    try:
      document = json.loads(body)
      if status == 200:
        ok = EnrolmentOk.from_wire(document)
        if ok.provider != request.provider or ok.risk_user_id != request.risk_user_id:
          raise ValueError("response identity mismatch")
        log.info("User key enrolment fingerprints=%s", tuple(key.fingerprint for key in ok.keys))
        return ok
      refusal = EnrolmentRefusal.from_wire(document)
    except ValueError:
      log.warning("User key service code=enrolment_response_invalid")
      raise UserKeyResolverUnavailable(request.provider, "enrolment_response_invalid") from None
    log.info("User key enrolment code=%s", refusal.error)
    if refusal.error == "identity_unresolved":
      raise IdentityUnresolved(request.provider)
    raise EnrolmentRefused(refusal)
