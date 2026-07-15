"""Authenticated risk control-plane client for live authority and invalidations."""

from __future__ import annotations

import hashlib
import hmac
import json
import secrets
import time
from typing import Callable
from urllib.parse import urlsplit
import ipaddress
from uuid import UUID

import httpx

from .commercial_authority_cache import CommercialAuthoritySnapshot


LIVE_PATH = "/internal/commercial/live-context"
AGREEMENT_TERMS_PATH = "/internal/commercial/agreement-terms"
FEED_PATH = "/internal/commercial/authority-invalidations"


class HttpCommercialAuthorityClient:
  def __init__(
    self,
    *,
    base_url: str,
    environment: str,
    key_id: str,
    secret: bytes,
    timeout_seconds: float = 5.0,
    clock: Callable[[], int] = lambda: int(time.time()),
    post: Callable[..., httpx.Response] = httpx.post,
  ) -> None:
    if environment not in {"dev", "staging", "prod"} or len(secret) < 32:
      raise ValueError("commercial authority client configuration is invalid")
    parsed_url = urlsplit(base_url)
    if (
      parsed_url.scheme not in {"http", "https"}
      or not parsed_url.hostname
      or parsed_url.username is not None
      or parsed_url.password is not None
      or parsed_url.query
      or parsed_url.fragment
    ):
      raise ValueError("commercial authority URL is invalid")
    if parsed_url.scheme != "https":
      try:
        loopback = parsed_url.hostname == "localhost" or ipaddress.ip_address(
          parsed_url.hostname
        ).is_loopback
      except ValueError:
        loopback = parsed_url.hostname == "localhost"
      if environment != "dev" or not loopback:
        raise ValueError("commercial authority transport requires HTTPS")
    self._base_url = base_url.rstrip("/")
    self.environment = environment
    self._key_id = key_id
    self._secret = secret
    self._timeout = timeout_seconds
    self._clock = clock
    self._post = post

  def load(self, context_id: UUID) -> CommercialAuthoritySnapshot:
    value = self._request(LIVE_PATH, {
      "environment": self.environment, "context_id": str(context_id),
    })
    if set(value) != {
      "environment", "context_id", "active", "entitlement_revision",
      "commercial_account_id", "mcp_token_id",
    }:
      raise ValueError("commercial authority snapshot schema is invalid")
    if value["environment"] != self.environment:
      raise ValueError("commercial authority environment is invalid")
    if type(value["active"]) is not bool or type(value["entitlement_revision"]) is not int:
      raise ValueError("commercial authority snapshot types are invalid")
    account_id = value["commercial_account_id"]
    if account_id is not None and type(account_id) is not int:
      raise ValueError("commercial authority account type is invalid")
    token_id = value["mcp_token_id"]
    if token_id is not None and not isinstance(token_id, str):
      raise ValueError("commercial authority token type is invalid")
    return CommercialAuthoritySnapshot(
      context_id=UUID(value["context_id"]),
      active=value["active"],
      entitlement_revision=value["entitlement_revision"],
      commercial_account_id=value.get("commercial_account_id"),
      token_id=UUID(value["mcp_token_id"]) if value.get("mcp_token_id") else None,
    )

  def fetch(self, after_sequence: int, *, limit: int = 100) -> dict:
    return self._request(FEED_PATH, {
      "after_sequence": after_sequence, "limit": limit,
    })

  def current_agreement_terms_revision(self, agreement_id: UUID) -> int | None:
    value = self._request(
      AGREEMENT_TERMS_PATH, {
        "environment": self.environment, "agreement_id": str(agreement_id),
      }
    )
    if set(value) != {
      "environment", "agreement_id", "current_terms_revision",
    }:
      raise ValueError("commercial agreement terms snapshot schema is invalid")
    if value["environment"] != self.environment:
      raise ValueError("commercial authority environment is invalid")
    try:
      returned_agreement_id = UUID(value["agreement_id"])
    except (TypeError, ValueError) as exc:
      raise ValueError("commercial agreement identity is invalid") from exc
    if returned_agreement_id != agreement_id:
      raise ValueError("commercial authority returned the wrong agreement")
    revision = value["current_terms_revision"]
    if revision is not None and (type(revision) is not int or revision <= 0):
      raise ValueError("commercial agreement terms revision is invalid")
    return revision

  def _request(self, path: str, document: dict) -> dict:
    body = json.dumps(document, separators=(",", ":"), sort_keys=True).encode()
    timestamp = str(self._clock())
    nonce = secrets.token_hex(16)
    digest = hashlib.sha256(body).hexdigest()
    message = f"POST\n{path}\n{timestamp}\n{nonce}\n{digest}".encode()
    signature = hmac.new(self._secret, message, hashlib.sha256).hexdigest()
    response = self._post(
      self._base_url + path,
      content=body,
      headers={
        "content-type": "application/json",
        "x-hank-service-key-id": self._key_id,
        "x-hank-request-timestamp": timestamp,
        "x-hank-request-nonce": nonce,
        "x-hank-request-signature": "v1=" + signature,
      },
      timeout=self._timeout,
    )
    response.raise_for_status()
    value = response.json()
    if not isinstance(value, dict):
      raise ValueError("commercial authority response is invalid")
    return value


__all__ = ["AGREEMENT_TERMS_PATH", "HttpCommercialAuthorityClient"]
