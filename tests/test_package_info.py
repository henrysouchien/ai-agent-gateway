import sys
from pathlib import Path

from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[3]
PKG_DIR = ROOT / "packages" / "agent-gateway"
if str(PKG_DIR) not in sys.path:
  sys.path.insert(0, str(PKG_DIR))

import agent_gateway
from agent_gateway.package_info import (
  CONTRACT_AUTONOMOUS_OPERATOR_MESSAGES_V1,
  CONTRACT_CONTROL_CHAT_CONTINUATION_V1,
  CONTRACT_CREDENTIAL_REFRESH_V1,
  PACKAGE_NAME,
)
from agent_gateway.server import ChatRuntime, GatewayServerConfig, create_gateway_app


class _NeverRunRunner:
  async def run(self, **_kwargs):
    raise AssertionError("runner should not be called by health checks")


async def _build_chat_runtime(_session, _request, _channel, _auth_manager):
  return ChatRuntime(
    system_prompt="test",
    build_runner=lambda _event_log, _sid: _NeverRunRunner(),
  )


def test_package_exports_contract_metadata() -> None:
  assert agent_gateway.PACKAGE_NAME == PACKAGE_NAME
  assert isinstance(agent_gateway.__version__, str)
  assert agent_gateway.__version__
  assert CONTRACT_CREDENTIAL_REFRESH_V1 in agent_gateway.CONTRACTS
  assert CONTRACT_AUTONOMOUS_OPERATOR_MESSAGES_V1 in agent_gateway.CONTRACTS
  assert CONTRACT_CONTROL_CHAT_CONTINUATION_V1 in agent_gateway.CONTRACTS
  assert agent_gateway.package_health()["contracts"] == sorted(agent_gateway.CONTRACTS)


def test_health_reports_package_contracts() -> None:
  app = create_gateway_app(GatewayServerConfig(build_chat_runtime=_build_chat_runtime))

  with TestClient(app) as client:
    response = client.get("/api/health")

  assert response.status_code == 200
  payload = response.json()
  assert payload["status"] == "ok"
  assert payload["package"]["name"] == PACKAGE_NAME
  assert CONTRACT_CREDENTIAL_REFRESH_V1 in payload["package"]["contracts"]
  assert CONTRACT_AUTONOMOUS_OPERATOR_MESSAGES_V1 in payload["package"]["contracts"]
  assert CONTRACT_CONTROL_CHAT_CONTINUATION_V1 in payload["package"]["contracts"]
