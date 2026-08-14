from __future__ import annotations

import ast
import os
from pathlib import Path
import threading

import pytest

from agent_gateway import claim_signing_authority as authority_module
from agent_gateway.claim_signing_authority import (
  GATEWAY_CLAIM_SIGNING_AUTHORITY_ENV,
  GATEWAY_CLAIM_SIGNING_AUTHORITY_FD_V1,
  GatewayClaimSigningAuthority,
  GatewayUserClaimSigner,
  LEGACY_CLAIM_HMAC_KEY_ENV,
  configured_gateway_claim_signing_authority,
  gateway_user_claim_signer_identity,
  install_gateway_claim_signing_authority,
  sign_gateway_user_claim,
)


_SECRET = b"claim-signing-authority-test-key-at-least-32-bytes"


class _ObservedEnvironment(dict[str, str]):
  def __init__(
    self,
    values: dict[str, str],
    *,
    fail_pop: str | None = None,
    fail_set_after_mutation: str | None = None,
  ) -> None:
    super().__init__(values)
    self.fail_pop = fail_pop
    self.fail_set_after_mutation = fail_set_after_mutation
    self.snapshots: list[tuple[str, dict[str, str]]] = []

  def pop(  # type: ignore[override]
    self,
    key: str,
    default: str | None = None,
  ) -> str | None:
    if key == self.fail_pop:
      self.snapshots.append((f"failed-pop:{key}", dict(self)))
      raise OSError(f"failed to remove {key}")
    value = super().pop(key, default)
    self.snapshots.append((f"pop:{key}", dict(self)))
    return value

  def __setitem__(self, key: str, value: str) -> None:
    super().__setitem__(key, value)
    self.snapshots.append((f"set:{key}", dict(self)))
    if key == self.fail_set_after_mutation:
      raise OSError(f"failed to publish {key}")


class _SameLayoutSignerSubstitution:
  __slots__ = ("__weakref__",)

  def sign_claim(self, *, ttl_seconds: int) -> dict[str, str]:
    _ = ttl_seconds
    return {"AGENT_API_CLAIM_USER_ID": "attacker"}


def _read_fd(payload: bytes) -> int:
  read_fd, write_fd = os.pipe()
  os.write(write_fd, payload)
  os.close(write_fd)
  return read_fd


def test_one_shot_fd_loader_closes_descriptor_and_signs(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  fd = _read_fd(_SECRET)
  os.set_inheritable(fd, True)
  inheritable_states: list[bool] = []
  original_set_inheritable = os.set_inheritable

  def record_set_inheritable(target_fd: int, inheritable: bool) -> None:
    assert target_fd == fd
    inheritable_states.append(inheritable)
    original_set_inheritable(target_fd, inheritable)

  monkeypatch.setattr(
    authority_module.os,
    "set_inheritable",
    record_set_inheritable,
  )
  authority = GatewayClaimSigningAuthority.from_one_shot_fd(fd)

  assert inheritable_states == [False]
  with pytest.raises(OSError):
    os.fstat(fd)
  claim = authority.sign_user_claim(
    user_id="42",
    user_email="owner@example.test",
    ttl_seconds=30,
  )
  assert claim["AGENT_API_CLAIM_USER_ID"] == "42"
  assert claim["AGENT_API_CLAIM_USER_EMAIL"] == "owner@example.test"


def test_bound_signer_is_immutable_secret_safe_and_stably_bound() -> None:
  authority = GatewayClaimSigningAuthority(_SECRET)
  signer = authority.bind_user(
    user_id="42",
    user_email="owner@example.test",
  )

  first = sign_gateway_user_claim(signer, ttl_seconds=30)
  with pytest.raises(AttributeError):
    setattr(signer, "_user_id", "attacker")
  with pytest.raises(AttributeError):
    setattr(signer, "_user_email", "attacker@example.test")
  with pytest.raises(AttributeError):
    object.__setattr__(signer, "_user_id", "attacker")
  with pytest.raises(AttributeError):
    object.__setattr__(
      signer,
      "_user_email",
      "attacker@example.test",
    )
  with pytest.raises(AttributeError):
    object.__setattr__(
      signer,
      "_GatewayUserClaimSigner__authority",
      GatewayClaimSigningAuthority(b"x" * 32),
    )
  binding = authority_module._gateway_user_claim_binding(signer)
  with pytest.raises(AttributeError):
    object.__setattr__(binding, "user_id", "attacker")
  with pytest.raises(RuntimeError, match="already bound"):
    signer.__init__(
      authority,
      user_id="attacker",
      user_email="attacker@example.test",
    )
  second = sign_gateway_user_claim(signer, ttl_seconds=30)

  for claim in (first, second):
    assert claim["AGENT_API_CLAIM_USER_ID"] == "42"
    assert claim["AGENT_API_CLAIM_USER_EMAIL"] == "owner@example.test"
  secret_text = _SECRET.decode("utf-8")
  assert secret_text not in repr(authority)
  assert secret_text not in repr(signer)
  assert "owner@example.test" not in repr(signer)


def test_same_layout_class_swap_cannot_reach_sealed_signing_operation() -> None:
  signer = GatewayClaimSigningAuthority(_SECRET).bind_user(
    user_id="42",
    user_email="owner@example.test",
  )

  object.__setattr__(
    signer,
    "__class__",
    _SameLayoutSignerSubstitution,
  )

  assert type(signer) is _SameLayoutSignerSubstitution
  assert signer.sign_claim(ttl_seconds=30) == {
    "AGENT_API_CLAIM_USER_ID": "attacker",
  }
  with pytest.raises(TypeError, match="type changed"):
    sign_gateway_user_claim(  # type: ignore[arg-type]
      signer,
      ttl_seconds=30,
    )
  with pytest.raises(TypeError, match="type changed"):
    gateway_user_claim_signer_identity(  # type: ignore[arg-type]
      signer,
    )


def test_gateway_signing_consumer_has_no_virtual_dispatch() -> None:
  assert "sign_claim" not in GatewayUserClaimSigner.__dict__
  runtime_path = (
    Path(__file__).resolve().parents[3]
    / "api/agent/autonomous/runtime_execution.py"
  )
  tree = ast.parse(runtime_path.read_text(encoding="utf-8"))
  calls = [
    node.func
    for node in ast.walk(tree)
    if isinstance(node, ast.Call)
  ]

  assert all(
    (
      isinstance(call.value, ast.Name)
      and call.value.id == "AutonomousClaimSigner"
    )
    for call in calls
    if isinstance(call, ast.Attribute)
    and call.attr == "sign_claim"
  )


@pytest.mark.parametrize(
  "payload",
  [
    b"short",
    _SECRET + b"\n",
    b"x" * 4097,
    b"x" * 32 + b"\x00",
  ],
)
def test_one_shot_fd_loader_rejects_noncanonical_key_and_closes(
  monkeypatch: pytest.MonkeyPatch,
  payload: bytes,
) -> None:
  fd = _read_fd(payload)
  os.set_inheritable(fd, True)
  inheritable_states: list[bool] = []
  original_set_inheritable = os.set_inheritable

  def record_set_inheritable(target_fd: int, inheritable: bool) -> None:
    assert target_fd == fd
    inheritable_states.append(inheritable)
    original_set_inheritable(target_fd, inheritable)

  monkeypatch.setattr(
    authority_module.os,
    "set_inheritable",
    record_set_inheritable,
  )
  with pytest.raises(ValueError):
    GatewayClaimSigningAuthority.from_one_shot_fd(fd)
  assert inheritable_states == [False]
  with pytest.raises(OSError):
    os.fstat(fd)


def test_install_scrubs_legacy_environment_value(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  monkeypatch.setattr(
    authority_module,
    "_configured_authority",
    None,
  )
  monkeypatch.setenv(LEGACY_CLAIM_HMAC_KEY_ENV, "must-disappear")
  authority = GatewayClaimSigningAuthority(_SECRET)

  install_gateway_claim_signing_authority(authority)

  assert configured_gateway_claim_signing_authority(
    required=True
  ) is authority
  assert LEGACY_CLAIM_HMAC_KEY_ENV not in os.environ
  assert (
    os.environ[GATEWAY_CLAIM_SIGNING_AUTHORITY_ENV]
    == GATEWAY_CLAIM_SIGNING_AUTHORITY_FD_V1
  )


def test_environment_publication_never_exposes_marker_with_legacy_secret(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  monkeypatch.setattr(
    authority_module,
    "_configured_authority",
    None,
  )
  environment = _ObservedEnvironment({
    GATEWAY_CLAIM_SIGNING_AUTHORITY_ENV: "stale",
    LEGACY_CLAIM_HMAC_KEY_ENV: "must-disappear",
  })
  monkeypatch.setattr(authority_module.os, "environ", environment)
  authority = GatewayClaimSigningAuthority(_SECRET)

  install_gateway_claim_signing_authority(authority)

  assert [
    operation for operation, _snapshot in environment.snapshots
  ] == [
    f"pop:{GATEWAY_CLAIM_SIGNING_AUTHORITY_ENV}",
    f"pop:{LEGACY_CLAIM_HMAC_KEY_ENV}",
    f"set:{GATEWAY_CLAIM_SIGNING_AUTHORITY_ENV}",
  ]
  assert all(
    not (
      GATEWAY_CLAIM_SIGNING_AUTHORITY_ENV in snapshot
      and LEGACY_CLAIM_HMAC_KEY_ENV in snapshot
    )
    for _operation, snapshot in environment.snapshots
  )
  assert environment == {
    GATEWAY_CLAIM_SIGNING_AUTHORITY_ENV: (
      GATEWAY_CLAIM_SIGNING_AUTHORITY_FD_V1
    ),
  }


def test_legacy_scrub_failure_leaves_marker_unpublished(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  monkeypatch.setattr(
    authority_module,
    "_configured_authority",
    None,
  )
  environment = _ObservedEnvironment(
    {
      GATEWAY_CLAIM_SIGNING_AUTHORITY_ENV: "stale",
      LEGACY_CLAIM_HMAC_KEY_ENV: "must-remain-unadvertised",
    },
    fail_pop=LEGACY_CLAIM_HMAC_KEY_ENV,
  )
  monkeypatch.setattr(authority_module.os, "environ", environment)
  authority = GatewayClaimSigningAuthority(_SECRET)

  with pytest.raises(OSError, match="failed to remove"):
    install_gateway_claim_signing_authority(authority)

  assert GATEWAY_CLAIM_SIGNING_AUTHORITY_ENV not in environment
  assert environment[LEGACY_CLAIM_HMAC_KEY_ENV] == (
    "must-remain-unadvertised"
  )
  assert configured_gateway_claim_signing_authority() is None


def test_partial_marker_write_is_rolled_back(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  monkeypatch.setattr(
    authority_module,
    "_configured_authority",
    None,
  )
  environment = _ObservedEnvironment(
    {
      GATEWAY_CLAIM_SIGNING_AUTHORITY_ENV: "stale",
      LEGACY_CLAIM_HMAC_KEY_ENV: "must-disappear",
    },
    fail_set_after_mutation=GATEWAY_CLAIM_SIGNING_AUTHORITY_ENV,
  )
  monkeypatch.setattr(authority_module.os, "environ", environment)
  authority = GatewayClaimSigningAuthority(_SECRET)

  with pytest.raises(OSError, match="failed to publish"):
    install_gateway_claim_signing_authority(authority)

  assert GATEWAY_CLAIM_SIGNING_AUTHORITY_ENV not in environment
  assert LEGACY_CLAIM_HMAC_KEY_ENV not in environment
  assert configured_gateway_claim_signing_authority() is None


def test_installation_publication_is_atomic_for_supported_readers(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  monkeypatch.setattr(
    authority_module,
    "_configured_authority",
    None,
  )
  authority = GatewayClaimSigningAuthority(_SECRET)
  environment_started = threading.Event()
  release_environment = threading.Event()
  reader_started = threading.Event()
  reader_finished = threading.Event()
  reader_result: list[GatewayClaimSigningAuthority | None] = []
  installer_failures: list[BaseException] = []

  def blocked_environment_publication() -> None:
    assert authority_module._configured_authority is None
    environment_started.set()
    if not release_environment.wait(timeout=5):
      raise RuntimeError("test environment publication timed out")
    monkeypatch.delenv(
      GATEWAY_CLAIM_SIGNING_AUTHORITY_ENV,
      raising=False,
    )
    monkeypatch.delenv(LEGACY_CLAIM_HMAC_KEY_ENV, raising=False)
    monkeypatch.setenv(
      GATEWAY_CLAIM_SIGNING_AUTHORITY_ENV,
      GATEWAY_CLAIM_SIGNING_AUTHORITY_FD_V1,
    )

  monkeypatch.setattr(
    authority_module,
    "_publish_claim_signing_authority_environment",
    blocked_environment_publication,
  )

  def install_authority() -> None:
    try:
      install_gateway_claim_signing_authority(authority)
    except BaseException as exc:
      installer_failures.append(exc)

  def read_authority() -> None:
    reader_started.set()
    reader_result.append(
      configured_gateway_claim_signing_authority()
    )
    reader_finished.set()

  installer = threading.Thread(target=install_authority)
  reader = threading.Thread(target=read_authority)
  installer.start()
  assert environment_started.wait(timeout=5)
  reader.start()
  assert reader_started.wait(timeout=5)
  assert not reader_finished.wait(timeout=0.05)
  release_environment.set()
  installer.join(timeout=5)
  reader.join(timeout=5)

  assert not installer.is_alive()
  assert not reader.is_alive()
  assert installer_failures == []
  assert reader_result == [authority]


def test_failed_environment_publication_does_not_install_authority(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  monkeypatch.setattr(
    authority_module,
    "_configured_authority",
    None,
  )
  authority = GatewayClaimSigningAuthority(_SECRET)

  def fail_environment_publication() -> None:
    raise OSError("environment unavailable")

  monkeypatch.setattr(
    authority_module,
    "_publish_claim_signing_authority_environment",
    fail_environment_publication,
  )

  with pytest.raises(OSError, match="environment unavailable"):
    install_gateway_claim_signing_authority(authority)

  assert configured_gateway_claim_signing_authority() is None
