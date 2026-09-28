from __future__ import annotations

import pytest

from agent_gateway.gateway_address import (
  DEFAULT_GATEWAY_BASE_URL,
  GATEWAY_BASE_URL_ENV,
  configured_gateway_base_url,
  gateway_base_url,
)


def test_unset_resolves_to_the_local_default() -> None:
  assert gateway_base_url({}) == DEFAULT_GATEWAY_BASE_URL
  assert configured_gateway_base_url({}) is None


def test_configured_origin_wins_and_is_normalized() -> None:
  environ = {GATEWAY_BASE_URL_ENV: "  https://localhost:8127/  "}

  assert configured_gateway_base_url(environ) == "https://localhost:8127"
  assert gateway_base_url(environ) == "https://localhost:8127"


def test_blank_is_not_configuration() -> None:
  environ = {GATEWAY_BASE_URL_ENV: "   "}

  assert configured_gateway_base_url(environ) is None
  assert gateway_base_url(environ) == DEFAULT_GATEWAY_BASE_URL


def test_a_configured_but_malformed_origin_is_not_silently_the_default() -> None:
  for raw in ("/", "///"):
    environ = {GATEWAY_BASE_URL_ENV: raw}

    assert configured_gateway_base_url(environ) == raw
    assert gateway_base_url(environ) == raw


def test_no_second_name_answers_for_the_gateway_origin() -> None:
  environ = {
    "GATEWAY_URL": "https://localhost:9999",
    "AGENTS_MCP_GATEWAY_URL": "https://localhost:9998",
  }

  assert configured_gateway_base_url(environ) is None
  assert gateway_base_url(environ) == DEFAULT_GATEWAY_BASE_URL


def test_process_environment_is_the_default_source(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  monkeypatch.setenv(GATEWAY_BASE_URL_ENV, "https://localhost:8127")

  assert gateway_base_url() == "https://localhost:8127"

  monkeypatch.delenv(GATEWAY_BASE_URL_ENV, raising=False)

  assert gateway_base_url() == DEFAULT_GATEWAY_BASE_URL


def test_autonomous_child_inherits_the_canonical_name_alone() -> None:
  from agent_gateway.autonomous_runner_start import (
    _AUTONOMOUS_CHILD_BASE_ENV_NAMES,
  )

  assert GATEWAY_BASE_URL_ENV in _AUTONOMOUS_CHILD_BASE_ENV_NAMES
  assert "GATEWAY_URL" not in _AUTONOMOUS_CHILD_BASE_ENV_NAMES
