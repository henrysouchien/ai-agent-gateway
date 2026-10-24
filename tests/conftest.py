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
def _pin_ollama_daemon_absent():
  """Every test sees the Ollama daemon absent, whatever runs on the host.

  ``OllamaLocalProvider.has_active_credential`` is a live ``GET /api/version`` probe
  cached per base for the process lifetime; the loopback origin is seeded absent (the
  repo-root ``conftest.py`` does the same for a root-collected run). A test whose subject
  needs a present daemon requests ``ollama_daemon_present``. The probe's own tests
  (``test_ollama_local_presence.py``) reset the cache and probe through their fake transports.
  """
  from agent_gateway.providers import ollama as ollama_provider

  ollama_provider._reset_presence_cache()
  ollama_provider._presence_cache[ollama_provider._DAEMON_ORIGIN] = False
  yield
  ollama_provider._reset_presence_cache()


@pytest.fixture
def ollama_daemon_present():
  """Opt in: the loopback Ollama daemon answers its presence probe."""
  from agent_gateway.providers import ollama as ollama_provider

  ollama_provider._presence_cache[ollama_provider._DAEMON_ORIGIN] = True


@pytest.fixture(autouse=True)
def _isolated_server_policy():
  from agent_gateway.policy_imports import configure_server_policy, load_server_policy_module

  previous = load_server_policy_module()
  configure_server_policy(None)
  try:
    yield
  finally:
    configure_server_policy(previous)