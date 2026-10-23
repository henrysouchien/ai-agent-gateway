from pathlib import Path
import sys

import pytest

PKG_DIR = Path(__file__).resolve().parents[1]
if str(PKG_DIR) not in sys.path:
  sys.path.insert(0, str(PKG_DIR))

from gateway_test_support.app_fixtures import (  # noqa: E402, F401
  auth_config_model_free,
  make_test_app,
)



@pytest.fixture(autouse=True)
def _isolated_gateway_state(monkeypatch, tmp_path_factory):
  state = tmp_path_factory.mktemp("package-state")
  (state / "gateway").mkdir(parents=True, mode=0o700)
  (state / "spend").mkdir(parents=True, mode=0o700)
  monkeypatch.setenv("USER_DATA_DIR", str(state))
  monkeypatch.setenv("AGENT_SESSION_LOG_BASE_DIR", str(state / "sessions"))
  monkeypatch.setenv("AGENT_GATEWAY_AUTONOMOUS_LOG_DIR", str(state / "autonomous"))
  monkeypatch.setenv("HANK_SPEND_LEDGER_PATH", str(state / "spend" / "api_spend.sqlite"))
  monkeypatch.setenv("HANK_SPEND_SETTINGS_PATH", str(state / "spend-settings.json"))
  monkeypatch.setenv("HANK_RUN_SETTINGS_PATH", str(state / "run-settings.json"))


@pytest.fixture(autouse=True)
def _isolated_server_policy():
  from agent_gateway.policy_imports import configure_server_policy, load_server_policy_module

  previous = load_server_policy_module()
  configure_server_policy(None)
  try:
    yield
  finally:
    configure_server_policy(previous)