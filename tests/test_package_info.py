# ruff: noqa: E402

import asyncio
import base64
import sys
import tomllib
from pathlib import Path

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

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
  CONTRACT_CHAT_ATTACHMENTS_V1,
  CONTRACT_INVESTMENT_SELECTED_CONTENT_V1,
  CONTRACT_CONTROL_CHAT_CONTINUATION_V1,
  CONTRACT_CONTROL_RUN_V1,
  CONTRACT_CREDENTIAL_REFRESH_V1,
  PACKAGE_NAME,
)
from agent_gateway.server import ChatRuntime, GatewayServerConfig, create_gateway_app
from agent_gateway.selected_content import SelectedContentAdmission
from agent_gateway.investment_capability_claim import (
  INVESTMENT_CAPABILITY_CLAIM_PRIVATE_KEY_ENV,
)


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
  assert CONTRACT_CONTROL_RUN_V1 in agent_gateway.CONTRACTS
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
  assert CONTRACT_CONTROL_RUN_V1 in payload["package"]["contracts"]
  assert CONTRACT_CHAT_ATTACHMENTS_V1 not in payload["package"]["contracts"]


def _selected_content_health_app(*, mcp_client=None):
  async def _admit(_session, _request):
    return SelectedContentAdmission()

  return create_gateway_app(
    GatewayServerConfig(
      tenant_id="test-product",
      model_registry=INITIAL_MODEL_REGISTRY,
      model_selection_policy=INITIAL_MODEL_SELECTION_POLICY,
      build_chat_runtime=_build_chat_runtime,
      selected_content_admitter=_admit,
      mcp_client=mcp_client,
    )
  )


class _InvestmentReaderMcp:
  def __init__(self) -> None:
    self.calls: list[tuple[str, dict]] = []

  @staticmethod
  def is_mcp_tool(name: str) -> bool:
    return name == "get_investment_artifact"

  @staticmethod
  def get_server_for_tool(name: str) -> str | None:
    return "idea-workbench-mcp" if name == "get_investment_artifact" else None

  async def call_tool(self, name: str, arguments: dict, **_kwargs):
    self.calls.append((name, arguments))
    return {"ok": False, "error": {"code": "artifact_gone"}}, None


def test_health_advertises_generic_selected_content_without_investment_reader() -> None:
  app = _selected_content_health_app()

  with TestClient(app) as client:
    response = client.get("/api/health")

  assert response.status_code == 200
  assert CONTRACT_CHAT_ATTACHMENTS_V1 in response.json()["package"]["contracts"]
  assert CONTRACT_INVESTMENT_SELECTED_CONTENT_V1 not in response.json()["package"]["contracts"]


def test_health_advertises_loaded_investment_reader_with_usable_signer_without_probing_it(
  monkeypatch,
) -> None:
  private_key = Ed25519PrivateKey.from_private_bytes(bytes(range(32)))
  key_material = base64.urlsafe_b64encode(private_key.private_bytes(
    encoding=serialization.Encoding.Raw,
    format=serialization.PrivateFormat.Raw,
    encryption_algorithm=serialization.NoEncryption(),
  )).rstrip(b"=").decode("ascii")
  monkeypatch.setenv(INVESTMENT_CAPABILITY_CLAIM_PRIVATE_KEY_ENV, key_material)
  mcp = _InvestmentReaderMcp()
  app = _selected_content_health_app(mcp_client=mcp)

  with TestClient(app) as client:
    first = client.get("/api/health")
    artifact_result, artifact_error = asyncio.run(mcp.call_tool(
      "get_investment_artifact",
      {"artifact_id": "artifact_missing", "view": "summary"},
    ))
    second = client.get("/api/health")

  assert first.status_code == 200
  assert second.status_code == 200
  assert CONTRACT_CHAT_ATTACHMENTS_V1 in first.json()["package"]["contracts"]
  assert CONTRACT_INVESTMENT_SELECTED_CONTENT_V1 in first.json()["package"]["contracts"]
  assert CONTRACT_INVESTMENT_SELECTED_CONTENT_V1 in second.json()["package"]["contracts"]
  assert artifact_result == {"ok": False, "error": {"code": "artifact_gone"}}
  assert artifact_error is None
  assert mcp.calls == [(
    "get_investment_artifact",
    {"artifact_id": "artifact_missing", "view": "summary"},
  )]


def test_health_omits_investment_contract_when_signer_is_unavailable(
  monkeypatch,
) -> None:
  monkeypatch.delenv(INVESTMENT_CAPABILITY_CLAIM_PRIVATE_KEY_ENV, raising=False)
  app = _selected_content_health_app(mcp_client=_InvestmentReaderMcp())

  with TestClient(app) as client:
    response = client.get("/api/health")

  assert response.status_code == 200
  assert response.json()["status"] == "ok"
  assert CONTRACT_CHAT_ATTACHMENTS_V1 in response.json()["package"]["contracts"]
  assert CONTRACT_INVESTMENT_SELECTED_CONTENT_V1 not in (
    response.json()["package"]["contracts"]
  )


def test_optional_investment_catalog_failure_does_not_fail_gateway_health() -> None:
  class _UnavailableCatalog:
    @staticmethod
    def is_mcp_tool(_name: str) -> bool:
      raise RuntimeError("catalog unavailable")

    @staticmethod
    def get_server_for_tool(_name: str) -> str | None:
      raise AssertionError("server lookup must stop after unavailable catalog")

  app = _selected_content_health_app(mcp_client=_UnavailableCatalog())

  with TestClient(app) as client:
    response = client.get("/api/health")

  assert response.status_code == 200
  assert response.json()["status"] == "ok"
  assert CONTRACT_CHAT_ATTACHMENTS_V1 in response.json()["package"]["contracts"]
  assert CONTRACT_INVESTMENT_SELECTED_CONTENT_V1 not in response.json()["package"]["contracts"]
