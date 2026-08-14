# ruff: noqa: E402

import sys
import tomllib
from pathlib import Path

from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[3]
PKG_DIR = ROOT / "packages" / "agent-gateway"
if str(PKG_DIR) not in sys.path:
  sys.path.insert(0, str(PKG_DIR))

import agent_gateway
from agent_gateway import package_info
from agent_gateway.model_registry import (
  INITIAL_MODEL_REGISTRY,
  INITIAL_MODEL_SELECTION_POLICY,
)
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


async def _build_chat_runtime(_session, request, _channel, _auth_manager):
  return ChatRuntime(
    system_prompt="test",
    build_runner=lambda _event_log, _sid, _started_at: _NeverRunRunner(),
    capability_execution=request.capability_execution,
  )


def test_package_exports_contract_metadata() -> None:
  assert agent_gateway.PACKAGE_NAME == PACKAGE_NAME
  assert isinstance(agent_gateway.__version__, str)
  assert agent_gateway.__version__
  assert CONTRACT_CREDENTIAL_REFRESH_V1 in agent_gateway.CONTRACTS
  assert CONTRACT_AUTONOMOUS_OPERATOR_MESSAGES_V1 in agent_gateway.CONTRACTS
  assert CONTRACT_CONTROL_CHAT_CONTINUATION_V1 in agent_gateway.CONTRACTS
  assert agent_gateway.package_health()["contracts"] == sorted(agent_gateway.CONTRACTS)


def test_package_metadata_does_not_require_excel_relay_capability() -> None:
  metadata = tomllib.loads(
    (PKG_DIR / "pyproject.toml").read_text(encoding="utf-8")
  )

  dependencies = metadata["project"]["dependencies"]
  assert not any(
    dependency.partition(";")[0].strip().lower().startswith("excel-mcp")
    for dependency in dependencies
  )


def test_package_health_reports_deployment_commit(monkeypatch) -> None:
  monkeypatch.setattr(package_info, "SOURCE_COMMIT", "e" * 40)

  health = package_info.package_health()

  assert health["source_commit"] == "e" * 40
  assert health["source_commit_provenance"] == "deployment_environment"


def test_package_health_allows_missing_local_deployment_commit(monkeypatch) -> None:
  monkeypatch.setattr(package_info, "SOURCE_COMMIT", None)

  health = package_info.package_health()

  assert health["source_commit"] is None
  assert health["source_commit_provenance"] is None


def test_health_reports_package_contracts() -> None:
  app = create_gateway_app(
    GatewayServerConfig(
      tenant_id="test-product",
      model_registry=INITIAL_MODEL_REGISTRY,
      model_selection_policy=INITIAL_MODEL_SELECTION_POLICY,
      build_chat_runtime=_build_chat_runtime,
    )
  )

  with TestClient(app) as client:
    response = client.get("/api/health")

  assert response.status_code == 200
  payload = response.json()
  assert payload["status"] == "ok"
  assert payload["package"]["name"] == PACKAGE_NAME
  assert CONTRACT_CREDENTIAL_REFRESH_V1 in payload["package"]["contracts"]
  assert CONTRACT_AUTONOMOUS_OPERATOR_MESSAGES_V1 in payload["package"]["contracts"]
  assert CONTRACT_CONTROL_CHAT_CONTINUATION_V1 in payload["package"]["contracts"]
