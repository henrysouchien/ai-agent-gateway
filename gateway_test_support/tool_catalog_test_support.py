from __future__ import annotations

from agent_gateway.session import GatewaySession


OWNER_GATEWAY_SESSION = GatewaySession(
  session_id="tool-catalog-test-owner",
  api_key_hash="test",
  created_at=1,
  expires_at=2,
  user_id="tool-catalog-test-owner",
  role="owner",
)


__all__ = ["OWNER_GATEWAY_SESSION"]
