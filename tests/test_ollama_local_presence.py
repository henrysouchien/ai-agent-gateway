from __future__ import annotations

from collections.abc import Callable, Iterator

import httpx
import pytest

from agent_gateway.providers.ollama import OllamaLocalProvider, _reset_presence_cache


@pytest.fixture(autouse=True)
def _clear_presence_cache() -> Iterator[None]:
  _reset_presence_cache()
  yield
  _reset_presence_cache()


def _config(
  base_url: str,
  handler: Callable[[httpx.Request], httpx.Response],
) -> dict[str, object]:
  return {"base_url": base_url, "_transport": httpx.MockTransport(handler)}


def test_reachable_daemon_is_active() -> None:
  def handler(request: httpx.Request) -> httpx.Response:
    assert request.method == "GET"
    assert str(request.url) == "http://127.0.0.1:11434/api/version"
    return httpx.Response(200, json={"version": "0.1.0"})

  assert OllamaLocalProvider().has_active_credential(
    _config("http://127.0.0.1:11434", handler)
  ) is True


def test_connect_error_is_inactive_and_does_not_raise() -> None:
  calls = 0

  def handler(request: httpx.Request) -> httpx.Response:
    nonlocal calls
    calls += 1
    raise httpx.ConnectError("connection refused", request=request)

  provider = OllamaLocalProvider()
  config = _config("http://127.0.0.1:11434", handler)

  assert provider.has_active_credential(config) is False
  assert provider.has_active_credential(config) is False
  assert calls == 1


def test_non_2xx_is_inactive() -> None:
  def handler(request: httpx.Request) -> httpx.Response:
    return httpx.Response(503, request=request)

  assert OllamaLocalProvider().has_active_credential(
    _config("http://127.0.0.1:11434", handler)
  ) is False


def test_probe_runs_once_per_base() -> None:
  calls: list[str] = []

  def handler(request: httpx.Request) -> httpx.Response:
    calls.append(str(request.url))
    return httpx.Response(200, json={"version": "0"})

  provider = OllamaLocalProvider()
  config = _config("http://127.0.0.1:11434", handler)

  assert provider.has_active_credential(config) is True
  assert provider.has_active_credential(config) is True
  assert calls == ["http://127.0.0.1:11434/api/version"]


def test_two_bases_are_probed_independently() -> None:
  calls: list[str] = []

  def handler(request: httpx.Request) -> httpx.Response:
    calls.append(str(request.url))
    return httpx.Response(200, json={"version": "0"})

  provider = OllamaLocalProvider()
  first = _config("http://127.0.0.1:11434", handler)
  second = _config("http://127.0.0.1:11435", handler)

  assert provider.has_active_credential(first) is True
  assert provider.has_active_credential(second) is True
  assert provider.has_active_credential(first) is True
  assert provider.has_active_credential(second) is True
  assert calls == [
    "http://127.0.0.1:11434/api/version",
    "http://127.0.0.1:11435/api/version",
  ]


@pytest.mark.parametrize("base_url", [
  "http://127.0.0.1:11434/v1",
  "http://127.0.0.1:11434/v1/",
])
def test_v1_suffix_is_stripped(base_url: str) -> None:
  seen: list[str] = []

  def handler(request: httpx.Request) -> httpx.Response:
    seen.append(str(request.url))
    return httpx.Response(200, json={"version": "0"})

  assert OllamaLocalProvider().has_active_credential(_config(base_url, handler)) is True
  assert seen == ["http://127.0.0.1:11434/api/version"]


def test_missing_base_url_probes_loopback_origin() -> None:
  seen: list[str] = []

  def handler(request: httpx.Request) -> httpx.Response:
    seen.append(str(request.url))
    return httpx.Response(200, json={"version": "0"})

  config: dict[str, object] = {"_transport": httpx.MockTransport(handler)}

  assert OllamaLocalProvider().has_active_credential(config) is True
  assert seen == ["http://127.0.0.1:11434/api/version"]


def test_probe_without_transport_uses_httpx_get(monkeypatch: pytest.MonkeyPatch) -> None:
  seen: list[tuple[str, object]] = []

  def fake_get(url: str, **kwargs: object) -> httpx.Response:
    seen.append((url, kwargs.get("timeout")))
    return httpx.Response(200, json={"version": "0"})

  monkeypatch.setattr("agent_gateway.providers.ollama.httpx.get", fake_get)

  assert OllamaLocalProvider().has_active_credential({}) is True
  assert seen == [("http://127.0.0.1:11434/api/version", 0.5)]
