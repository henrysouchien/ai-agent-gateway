from __future__ import annotations

import ast
import gc
import hashlib
import hmac
import os
from pathlib import Path
import threading
from typing import cast

import pytest

from agent_gateway import claim_signing_authority as authority_module
from agent_gateway.autonomous_runner_claims import get_agent_api_claim_ttl_seconds
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
from agent_gateway.ttl_policy import AGENT_API_CLAIM_TTL


_SECRET = b"claim-signing-authority-test-key-at-least-32-bytes"


def test_native_producer_body_mac_binds_domain_key_and_exact_request() -> None:
  authority = GatewayClaimSigningAuthority(_SECRET)
  body = b'{"action":"begin","cell_token":"opaque","operation":"mcp__risk__positions"}'
  signature = hmac.new(
    _SECRET, b"hank/native-risk-cell/v1\0" + body, hashlib.sha256,
  ).hexdigest()
  assert authority.verify_native_risk_request(body, signature)
  assert not authority.verify_native_risk_request(body.replace(b"positions", b"trades"), signature)
  assert not authority.verify_native_risk_request(body + b" ", signature)
  foreign_domain = hmac.new(
    _SECRET, b"hank/user-claim/v1\0" + body, hashlib.sha256,
  ).hexdigest()
  assert not authority.verify_native_risk_request(body, foreign_domain)
  other_key = hmac.new(
    b"another-producer-key", b"hank/native-risk-cell/v1\0" + body, hashlib.sha256,
  ).hexdigest()
  assert not authority.verify_native_risk_request(body, other_key)
  assert not authority.verify_native_risk_request(body, "not-hex")


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



def _claim_ttl(claim: dict[str, str]) -> int:
  return (
    int(claim["AGENT_API_CLAIM_EXPIRY"])
    - int(claim["AGENT_API_CLAIM_ISSUED_AT"])
  )


def test_signing_resolves_each_users_entitlement_once(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  monkeypatch.delenv("AGENT_API_CLAIM_TTL_SECONDS", raising=False)
  records = {"alice": 90, "bob": 180}
  calls: list[tuple[str, str]] = []

  def resolver(user_id: str, ttl_name: str) -> int | None:
    calls.append((user_id, ttl_name))
    return records.get(user_id)

  authority = GatewayClaimSigningAuthority(_SECRET, ttl_resolver=resolver)
  for user_id, expected in (("alice", 90), ("bob", 180), ("missing", 300)):
    signer = authority.bind_user(user_id=user_id, user_email=None)
    claim = sign_gateway_user_claim(signer)
    assert claim["AGENT_API_CLAIM_USER_ID"] == user_id
    assert _claim_ttl(claim) == expected
  assert calls == [
    ("alice", AGENT_API_CLAIM_TTL),
    ("bob", AGENT_API_CLAIM_TTL),
    ("missing", AGENT_API_CLAIM_TTL),
  ]


def test_signing_preserves_environment_and_explicit_ttl_fallbacks(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  monkeypatch.setenv("AGENT_API_CLAIM_TTL_SECONDS", "432")
  for resolver in (None, lambda user_id, ttl_name: None):
    authority = GatewayClaimSigningAuthority(_SECRET, ttl_resolver=resolver)
    assert _claim_ttl(authority.sign_user_claim(
      user_id="missing", user_email=None,
    )) == 432
    assert _claim_ttl(authority.sign_user_claim(
      user_id="missing", user_email=None, ttl_seconds=45,
    )) == 45
    with pytest.raises(ValueError, match="must be positive"):
      authority.sign_user_claim(
        user_id="missing", user_email=None, ttl_seconds=0,
      )
  monkeypatch.delenv("AGENT_API_CLAIM_TTL_SECONDS")
  assert _claim_ttl(GatewayClaimSigningAuthority(_SECRET).sign_user_claim(
    user_id="missing", user_email=None,
  )) == 300


def test_entitlement_ttl_overrides_environment_and_explicit_fallback(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  monkeypatch.setenv("AGENT_API_CLAIM_TTL_SECONDS", "432")
  authority = GatewayClaimSigningAuthority(
    _SECRET, ttl_resolver=lambda user_id, ttl_name: 90,
  )
  assert get_agent_api_claim_ttl_seconds(
    "alice", lambda user_id, ttl_name: 90,
  ) == 90
  assert _claim_ttl(authority.sign_user_claim(
    user_id="alice", user_email=None,
  )) == 90
  assert _claim_ttl(authority.sign_user_claim(
    user_id="alice", user_email=None, ttl_seconds=45,
  )) == 90


@pytest.mark.parametrize("resolved", (None, True, False, "90", 90.5, 0, -1))
def test_invalid_entitlement_ttl_retains_fallback_at_helper_and_signing(
  monkeypatch: pytest.MonkeyPatch,
  resolved: object,
) -> None:
  def resolver(user_id: str, ttl_name: str) -> int | None:
    return cast(int | None, resolved)

  authority = GatewayClaimSigningAuthority(_SECRET, ttl_resolver=resolver)
  for environment, expected in (("432", 432), ("", 300)):
    monkeypatch.setenv("AGENT_API_CLAIM_TTL_SECONDS", environment)
    assert get_agent_api_claim_ttl_seconds("alice", resolver) == expected
    assert _claim_ttl(authority.sign_user_claim(
      user_id="alice", user_email=None,
    )) == expected
    assert _claim_ttl(authority.sign_user_claim(
      user_id="alice", user_email=None, ttl_seconds=45,
    )) == 45


@pytest.mark.parametrize(
  ("environment", "expected"),
  (("", 300), ("not-an-int", 300), ("0", 300), ("-1", 300), (" 73 ", 73)),
)
def test_claim_ttl_helper_preserves_environment_fallback(
  monkeypatch: pytest.MonkeyPatch,
  environment: str,
  expected: int,
) -> None:
  monkeypatch.setenv("AGENT_API_CLAIM_TTL_SECONDS", environment)
  assert get_agent_api_claim_ttl_seconds() == expected
  assert get_agent_api_claim_ttl_seconds(
    "alice", lambda user_id, ttl_name: None,
  ) == expected


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
    sign_gateway_user_claim(
      signer,
      ttl_seconds=30,
    )
  with pytest.raises(TypeError, match="type changed"):
    gateway_user_claim_signer_identity(
      signer,
    )


class _CollectingLock:
  """Runs a full collection each time the binding table's lock is taken."""

  def __init__(self) -> None:
    self._lock = threading.Lock()

  def __enter__(self) -> None:
    self._lock.acquire()
    gc.collect()

  def __exit__(self, *exc_info: object) -> None:
    self._lock.release()


def test_collection_inside_binding_section_does_not_wait_on_own_lock(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  monkeypatch.setattr(
    authority_module,
    "_gateway_user_claim_bindings_lock",
    _CollectingLock(),
  )
  authority = GatewayClaimSigningAuthority(_SECRET)
  gc_was_enabled = gc.isenabled()
  gc.disable()
  try:
    dead = authority.bind_user(user_id="dead", user_email=None)
    cycle: list[object] = [dead]
    cycle.append(cycle)
    del dead, cycle
    bound: list[tuple[str, str | None]] = []

    def bind_next() -> None:
      signer = authority.bind_user(
        user_id="next",
        user_email="next@example.test",
      )
      bound.append(gateway_user_claim_signer_identity(signer))

    worker = threading.Thread(target=bind_next, daemon=True)
    worker.start()
    worker.join(timeout=10)
  finally:
    if gc_was_enabled:
      gc.enable()

  assert not worker.is_alive(), "binding thread waited on its own lock"
  assert bound == [("next", "next@example.test")]




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
