from __future__ import annotations

import httpx2
from hank_contracts.agent_api import AGENT_API_CLAIM_AUDIENCE, AGENT_API_CLAIM_ENV_VARS


class GatewayClaimMintError(RuntimeError):
  """A gateway claim request failed without exposing request or response secrets."""

  def __init__(self, message: str, *, status_code: int | None = None) -> None:
    super().__init__(message)
    self.status_code = status_code


class GatewayClaimMintSchemaError(GatewayClaimMintError):
  """The gateway did not return the claim environment wire contract."""


def _claim_environment(payload: object, *, status_code: int) -> dict[str, str]:
  if (
    type(payload) is not dict
    or set(payload) != set(AGENT_API_CLAIM_ENV_VARS.values())
    or any(type(value) is not str for value in payload.values())
  ):
    raise GatewayClaimMintSchemaError(
      "gateway claim mint response schema is invalid", status_code=status_code,
    ) from None
  if payload[AGENT_API_CLAIM_ENV_VARS["audience"]] != AGENT_API_CLAIM_AUDIENCE:
    raise GatewayClaimMintSchemaError(
      "gateway claim mint response schema is invalid", status_code=status_code,
    ) from None
  return payload


async def _request_claim(
  client: httpx2.AsyncClient, *, url: str, api_key: str,
) -> dict[str, str]:
  try:
    response = await client.post(
      url,
      headers={"Authorization": f"Bearer {api_key}"},
      auth=httpx2.Auth(),
      follow_redirects=False,
    )
  except (httpx2.HTTPError, httpx2.InvalidURL):
    raise GatewayClaimMintError("gateway claim mint request failed") from None
  if not response.is_success:
    raise GatewayClaimMintError(
      "gateway claim mint request was rejected", status_code=response.status_code,
    ) from None
  try:
    payload = response.json()
  except (ValueError, UnicodeError):
    raise GatewayClaimMintSchemaError(
      "gateway claim mint response is not valid JSON", status_code=response.status_code,
    ) from None
  return _claim_environment(payload, status_code=response.status_code)

def _request_claim_sync(
  client: httpx2.Client, *, url: str, api_key: str,
) -> dict[str, str]:
  try:
    response = client.post(
      url,
      headers={"Authorization": f"Bearer {api_key}"},
      auth=httpx2.Auth(),
      follow_redirects=False,
    )
  except (httpx2.HTTPError, httpx2.InvalidURL):
    raise GatewayClaimMintError("gateway claim mint request failed") from None
  if not response.is_success:
    raise GatewayClaimMintError(
      "gateway claim mint request was rejected", status_code=response.status_code,
    ) from None
  try:
    payload = response.json()
  except (ValueError, UnicodeError):
    raise GatewayClaimMintSchemaError(
      "gateway claim mint response is not valid JSON", status_code=response.status_code,
    ) from None
  return _claim_environment(payload, status_code=response.status_code)


def mint_gateway_user_claim_sync(
  *,
  base_url: str,
  api_key: str,
  client: httpx2.Client | None = None,
) -> dict[str, str]:
  """Mint synchronously for local execution paths that cannot await."""
  if not base_url.strip():
    raise ValueError("gateway base URL must not be empty")
  url = f"{base_url.rstrip('/')}/api/claims/mint"
  if client is not None:
    return _request_claim_sync(client, url=url, api_key=api_key)
  with httpx2.Client(timeout=None, follow_redirects=False) as owned_client:
    return _request_claim_sync(owned_client, url=url, api_key=api_key)


async def mint_gateway_user_claim(
  *,
  base_url: str,
  api_key: str,
  client: httpx2.AsyncClient | None = None,
) -> dict[str, str]:
  """Mint the roster bearer's own claim; callers cannot select an identity."""
  if not base_url.strip():
    raise ValueError("gateway base URL must not be empty")
  url = f"{base_url.rstrip('/')}/api/claims/mint"
  if client is not None:
    return await _request_claim(client, url=url, api_key=api_key)
  async with httpx2.AsyncClient(timeout=None, follow_redirects=False) as owned_client:
    return await _request_claim(owned_client, url=url, api_key=api_key)
