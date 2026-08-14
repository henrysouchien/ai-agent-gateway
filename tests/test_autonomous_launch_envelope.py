from __future__ import annotations

import hashlib
import hmac
import json
from pathlib import Path
from typing import Any

import pytest

import agent_gateway.autonomous_launch_envelope as envelope_module
from agent_gateway.autonomous_launch_envelope import (
  AUTONOMOUS_CAPABILITY_ENVELOPE_AUDIENCE,
  AUTONOMOUS_CAPABILITY_ENVELOPE_VERSION,
  AUTONOMOUS_RUNTIME_SESSION_PURPOSE,
  AutonomousControlAuthority,
  AutonomousDispatchScope,
  AutonomousLaunchWorkload,
  AutonomousSessionAuthority,
  OrdinaryAutonomousSessionAuthority,
  sign_autonomous_launch_envelope,
  verify_autonomous_launch_envelope,
)
from agent_gateway.capability_binding import (
  CapabilityBind,
  CredentialHandle,
)
from agent_gateway.session import GatewaySession


_SECRET = "launch-envelope-test-secret-at-least-32-bytes"
_NOW_NS = 1_800_000_000_000_000_000
_NONCE = "0123456789abcdef0123456789abcdef"
_CHANNEL_ID = "12" * 32


def _control_authority() -> AutonomousControlAuthority:
  return AutonomousControlAuthority(
    control_mode="file",
    admission_ledger_path="/tmp/autonomous-admissions.sqlite3",
    admission_ledger_device=1,
    admission_ledger_inode=10,
    operator_inbox_path="/tmp/bg_7.operator-messages.jsonl",
    operator_inbox_device=1,
    operator_inbox_inode=11,
    approval_decisions_path="/tmp/bg_7.approval-decisions.jsonl",
    approval_decisions_device=1,
    approval_decisions_inode=12,
    approval_store_path="/tmp/autonomous-approvals.sqlite3",
    approval_store_device=1,
    approval_store_inode=13,
  )


def _memory_control_authority() -> AutonomousControlAuthority:
  return AutonomousControlAuthority(
    control_mode="memory",
    admission_ledger_path=None,
    admission_ledger_device=None,
    admission_ledger_inode=None,
    operator_inbox_path=None,
    operator_inbox_device=None,
    operator_inbox_inode=None,
    approval_decisions_path=None,
    approval_decisions_device=None,
    approval_decisions_inode=None,
    approval_store_path=None,
    approval_store_device=None,
    approval_store_inode=None,
  )


def _workload(
  *,
  profile: str = "analyst",
  mode: str = "run_once",
  task: str | None = None,
  skill: str | None = None,
  pack: str | None = None,
  context: str | None = None,
  ticker: str | None = None,
  dev_mode: bool = False,
  max_budget_usd: float | None = None,
  deliver: bool = True,
) -> AutonomousLaunchWorkload:
  return AutonomousLaunchWorkload(
    profile=profile,
    mode=mode,  # type: ignore[arg-type]
    task=task,
    skill=skill,
    pack=pack,
    context=context,
    ticker=ticker,
    dev_mode=dev_mode,
    max_budget_usd=max_budget_usd,
    deliver=deliver,
  )


def _bind(
  *,
  principal: str = "service",
  run_mode: str = "autonomous",
) -> CapabilityBind:
  return CapabilityBind(
    schema_version="1.0",
    capability_id="session.driver",
    model_key="anthropic.test-autonomous",
    provider="anthropic",
    upstream_model="claude-test",
    adapter="anthropic.messages",
    protocol_profile="messages.standard",
    route="anthropic.public",
    effort="high",
    credential_principal=principal,  # type: ignore[arg-type]
    credential_ref=(
      "autonomous-user:test-handle"
      if principal == "user"
      else "autonomous-service:test-handle"
    ),
    run_mode=run_mode,  # type: ignore[arg-type]
    registry_revision="test-registry-1",
    policy_revision="test-policy-1",
    selection_source="capability_default",
  )


def _credential_handle(
  *,
  tenant_id: str = "tenant-ordinary",
  principal: str = "user",
) -> CredentialHandle:
  return CredentialHandle(
    handle_id=(
      "autonomous-user:test-handle"
      if principal == "user"
      else "autonomous-service:test-handle"
    ),
    provider="anthropic",
    principal=principal,  # type: ignore[arg-type]
    tenant_id=tenant_id,
    actor_id="42" if principal == "user" else None,
  )


def _ordinary_session_authority(
  *,
  bind: CapabilityBind | None = None,
  dispatch_scope: AutonomousDispatchScope | None = None,
  role: object = "owner",
) -> AutonomousSessionAuthority:
  resolved_bind = bind or _bind()
  handle = _credential_handle(
    tenant_id="tenant-ordinary",
    principal=resolved_bind.credential_principal,
  )
  return AutonomousSessionAuthority.ordinary(
    OrdinaryAutonomousSessionAuthority(
      session_id="bg_7",
      tenant_id="tenant-ordinary",
      user_id="42",
      owner_user_id="42",
      created_at=1_799_999_900,
      expires_at=1_800_000_600,
      user_email="owner@example.test",
      risk_user_id=7,
      role=role,  # type: ignore[arg-type]
      kind="chat",
      channel="cli",
      purpose=AUTONOMOUS_RUNTIME_SESSION_PURPOSE,
      raw_user_id="42",
      user_slug="owner",
      user_aliases=("42", "owner", "owner@example.test"),
      identity_status="canonical",
      schema_version=1,
      is_public=False,
      allow_service_for_interactive=False,
      auth_provider=resolved_bind.provider,
      credential_handle=handle,
    ),
    dispatch_scope=dispatch_scope,
  )


def test_ordinary_authority_accepts_invite_role_exactly() -> None:
  session = _ordinary_session_authority(role="invite").to_gateway_session()
  assert session.role == "invite"


@pytest.mark.parametrize("role", ["Owner", " owner ", "OWNER", "", None, True])
def test_ordinary_authority_rejects_malformed_role(role: object) -> None:
  with pytest.raises(ValueError, match="role must be exactly"):
    _ordinary_session_authority(role=role)


def _signed(**overrides: object) -> str:
  bind = overrides.get("bind", _bind())
  assert isinstance(bind, CapabilityBind)
  if "session_authority" in overrides:
    session_authority = overrides["session_authority"]
  else:
    session_authority = _ordinary_session_authority(bind=bind)
  kwargs: dict[str, object] = {
    "task_id": "bg_7",
    "control_run_id": "run-7",
    "owner_user_id": "42",
    "channel_id": _CHANNEL_ID,
    "bind": bind,
    "workload": _workload(),
    "control_authority": _control_authority(),
    "session_authority": session_authority,
    "ttl_seconds": 60,
    "now_ns": _NOW_NS,
    "nonce": _NONCE,
  }
  kwargs.update(overrides)
  return sign_autonomous_launch_envelope(  # type: ignore[arg-type]
    _SECRET,
    **kwargs,
  )


def _verify(envelope_json: str, **overrides: object):
  kwargs: dict[str, object] = {"now_ns": _NOW_NS}
  kwargs.update(overrides)
  return verify_autonomous_launch_envelope(  # type: ignore[arg-type]
    _SECRET,
    envelope_json,
    **kwargs,
  )


def _resign(payload: dict[str, object]) -> str:
  unsigned = dict(payload)
  unsigned.pop("signature", None)
  canonical = json.dumps(
    unsigned,
    sort_keys=True,
    separators=(",", ":"),
    ensure_ascii=False,
  )
  payload["signature"] = hmac.new(
    _SECRET.encode("utf-8"),
    canonical.encode("utf-8"),
    hashlib.sha256,
  ).hexdigest()
  return json.dumps(
    payload,
    sort_keys=True,
    separators=(",", ":"),
    ensure_ascii=False,
  )


def test_v4_round_trip_constructs_exact_gateway_session() -> None:
  dispatch_scope = AutonomousDispatchScope(
    kind="portfolio",
    source="user_selected",
    portfolio_name="Core",
    portfolio_id="portfolio-7",
    display_name="Core Portfolio",
  )
  authority = _ordinary_session_authority(
    dispatch_scope=dispatch_scope
  )
  raw = _signed(session_authority=authority)
  parsed = json.loads(raw)

  assert raw == json.dumps(
    parsed,
    sort_keys=True,
    separators=(",", ":"),
    ensure_ascii=False,
  )
  assert _SECRET not in raw

  envelope = _verify(raw)
  assert envelope.audience == AUTONOMOUS_CAPABILITY_ENVELOPE_AUDIENCE
  assert envelope.version == AUTONOMOUS_CAPABILITY_ENVELOPE_VERSION == 4
  assert envelope.task_id == "bg_7"
  assert envelope.control_run_id == "run-7"
  assert envelope.owner_user_id == "42"
  assert envelope.channel_id == _CHANNEL_ID
  assert envelope.bind == _bind()
  assert envelope.workload == _workload()
  assert envelope.control_authority == _control_authority()
  assert envelope.session_authority == authority

  session = envelope.session_authority.to_gateway_session()
  assert type(session) is GatewaySession
  assert session.session_id == "bg_7"
  assert session.user_id == "42"
  assert session.owner_user_id == "42"
  assert session.purpose == AUTONOMOUS_RUNTIME_SESSION_PURPOSE
  assert session.dispatch_scope == dispatch_scope.receipt()


def test_envelope_rejects_tampering_and_wrong_signature() -> None:
  payload = json.loads(_signed())
  payload["workload"]["profile"] = "advisor"

  with pytest.raises(ValueError, match="signature is invalid"):
    _verify(
      json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
      )
    )
  with pytest.raises(ValueError, match="signature is invalid"):
    verify_autonomous_launch_envelope(
      "wrong-launch-envelope-secret-at-least-32-bytes",
      _signed(),
      now_ns=_NOW_NS,
    )


def test_envelope_rejects_hmac_secrets_shorter_than_32_bytes() -> None:
  with pytest.raises(ValueError, match="at least 32 bytes"):
    sign_autonomous_launch_envelope(
      "short-secret",
      task_id="bg_7",
      control_run_id="run-7",
      owner_user_id="42",
      channel_id=_CHANNEL_ID,
      bind=_bind(),
      workload=_workload(),
      control_authority=_control_authority(),
      session_authority=_ordinary_session_authority(),
      now_ns=_NOW_NS,
      nonce=_NONCE,
    )

  with pytest.raises(ValueError, match="at least 32 bytes"):
    verify_autonomous_launch_envelope(
      b"too-short",
      _signed(),
      now_ns=_NOW_NS,
    )


@pytest.mark.parametrize(
  ("mutation", "message"),
  [
    (lambda payload: payload.pop("task_id"), "missing fields: task_id"),
    (
      lambda payload: payload.__setitem__("provider", "anthropic"),
      "unexpected fields: provider",
    ),
    (
      lambda payload: payload.__setitem__("version", 2),
      "version is unsupported",
    ),
    (
      lambda payload: payload.__setitem__(
        "audience",
        "wrong-audience",
      ),
      "audience is invalid",
    ),
  ],
)
def test_envelope_rejects_closed_contract_changes(
  mutation: Any,
  message: str,
) -> None:
  payload = json.loads(_signed())
  mutation(payload)

  with pytest.raises(ValueError, match=message):
    _verify(_resign(payload))


@pytest.mark.parametrize(
  "workload",
  [
    _workload(),
    _workload(
      mode="task",
      task="Investigate the variance",
    ),
    _workload(
      mode="pack",
      pack="daily-risk-pack",
    ),
    _workload(
      mode="skill",
      skill="earnings-review",
      context="Quarterly review\nUse filed results.",
      ticker="MSFT",
      dev_mode=True,
      max_budget_usd=12.5,
      deliver=False,
    ),
  ],
)
def test_envelope_round_trips_every_closed_workload_mode(
  workload: AutonomousLaunchWorkload,
) -> None:
  envelope = _verify(_signed(workload=workload))

  assert envelope.workload is not workload
  assert envelope.workload == workload
  assert envelope.payload()["workload"] == workload.receipt()


@pytest.mark.parametrize(
  "kwargs",
  [
    {"mode": "run_once", "skill": "earnings-review"},
    {"mode": "task", "task": "review", "dev_mode": True},
    {"mode": "task", "task": None},
    {"mode": "pack", "pack": "daily", "deliver": False},
    {"mode": "skill", "skill": None},
    {"mode": "skill", "skill": "review", "task": "other"},
    {"mode": "skill", "skill": "review", "max_budget_usd": True},
    {"mode": "skill", "skill": "review", "max_budget_usd": 12},
    {"mode": "skill", "skill": "review", "max_budget_usd": 0},
    {
      "mode": "skill",
      "skill": "review",
      "max_budget_usd": float("inf"),
    },
  ],
)
def test_workload_rejects_incompatible_or_noncanonical_fields(
  kwargs: dict[str, object],
) -> None:
  with pytest.raises(ValueError, match="autonomous launch workload"):
    _workload(**kwargs)  # type: ignore[arg-type]


def test_workload_free_text_limit_is_measured_in_utf8_bytes() -> None:
  with pytest.raises(
    ValueError,
    match="autonomous launch workload context is invalid",
  ):
    _workload(
      mode="skill",
      skill="review",
      context="\U0001f642" * (64 * 1024),
    )


@pytest.mark.parametrize("mutation", ["missing", "extra"])
def test_envelope_rejects_non_exact_workload_receipt(
  mutation: str,
) -> None:
  payload = json.loads(_signed())
  if mutation == "missing":
    payload["workload"].pop("deliver")
  else:
    payload["workload"]["command"] = "untrusted"

  with pytest.raises(ValueError, match="workload is invalid"):
    _verify(_resign(payload))


def test_envelope_signature_binds_the_exact_workload() -> None:
  payload = json.loads(_signed())
  payload["workload"]["profile"] = "advisor"

  with pytest.raises(ValueError, match="signature is invalid"):
    _verify(
      json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
      )
    )


@pytest.mark.parametrize(
  "kwargs",
  [
    {"operator_inbox_path": "relative.jsonl"},
    {"approval_decisions_path": "/tmp/../tmp/decisions.jsonl"},
    {
      "approval_store_path": (
        "/tmp/bg_7.operator-messages.jsonl"
      ),
    },
  ],
)
def test_control_authority_rejects_noncanonical_or_aliased_paths(
  kwargs: dict[str, object],
) -> None:
  values: dict[str, object] = {
    "control_mode": "file",
    "admission_ledger_path": "/tmp/autonomous-admissions.sqlite3",
    "admission_ledger_device": 1,
    "admission_ledger_inode": 10,
    "operator_inbox_path": "/tmp/bg_7.operator-messages.jsonl",
    "operator_inbox_device": 1,
    "operator_inbox_inode": 11,
    "approval_decisions_path": "/tmp/bg_7.approval-decisions.jsonl",
    "approval_decisions_device": 1,
    "approval_decisions_inode": 12,
    "approval_store_path": "/tmp/autonomous-approvals.sqlite3",
    "approval_store_device": 1,
    "approval_store_inode": 13,
  }
  values.update(kwargs)

  with pytest.raises(ValueError, match="autonomous control authority"):
    AutonomousControlAuthority(**values)  # type: ignore[arg-type]


@pytest.mark.parametrize("mutation", ["missing", "extra"])
def test_envelope_rejects_non_exact_control_authority_receipt(
  mutation: str,
) -> None:
  payload = json.loads(_signed())
  if mutation == "missing":
    payload["control_authority"].pop("operator_inbox_path")
  else:
    payload["control_authority"]["socket_fd"] = 7

  with pytest.raises(
    ValueError,
    match="control_authority is invalid",
  ):
    _verify(_resign(payload))


def test_envelope_signature_binds_the_exact_control_authority() -> None:
  payload = json.loads(_signed())
  payload["control_authority"]["operator_inbox_path"] = (
    "/tmp/cross-wired.operator-messages.jsonl"
  )

  with pytest.raises(ValueError, match="signature is invalid"):
    _verify(
      json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
      )
    )


def test_memory_control_authority_round_trips_without_files() -> None:
  authority = _memory_control_authority()

  assert AutonomousControlAuthority.from_receipt(
    authority.receipt()
  ) == authority
  assert all(
    value is None
    for field_name, value in authority.receipt().items()
    if field_name != "control_mode"
  )


def test_signing_rejects_process_local_memory_authority() -> None:
  with pytest.raises(
    ValueError,
    match="cannot cross a process boundary",
  ):
    _signed(control_authority=_memory_control_authority())


def test_verification_rejects_resigned_memory_authority() -> None:
  payload = json.loads(_signed())
  payload["control_authority"] = _memory_control_authority().receipt()

  with pytest.raises(
    ValueError,
    match="control_authority is invalid",
  ):
    _verify(_resign(payload))


@pytest.mark.parametrize(
  ("field_name", "value"),
  (
    ("created_at", 1_800_000_001),
    ("expires_at", 1_800_000_030),
  ),
)
def test_ordinary_session_lifetime_must_cover_envelope(
  field_name: str,
  value: int,
) -> None:
  payload = json.loads(_signed())
  payload["session_authority"]["ordinary_authority"][
    field_name
  ] = value

  with pytest.raises(
    ValueError,
    match="session authority lifetime",
  ):
    _verify(_resign(payload))


def test_envelope_rejects_duplicate_json_keys_at_every_depth() -> None:
  raw = _signed()
  duplicate = raw[:-1] + ',"task_id":"bg_7"}'
  with pytest.raises(ValueError, match="duplicate field: task_id"):
    _verify(duplicate)

  nested = _signed().replace(
    '"profile":"analyst"',
    '"profile":"analyst","profile":"analyst"',
  )
  with pytest.raises(ValueError, match="duplicate field: profile"):
    _verify(nested)

  with pytest.raises(ValueError, match="canonical JSON"):
    _verify(json.dumps(json.loads(raw), indent=2))


@pytest.mark.parametrize("mutation", ["missing", "extra"])
def test_envelope_rejects_non_exact_bind_receipt(
  mutation: str,
) -> None:
  payload = json.loads(_signed())
  if mutation == "missing":
    payload["capability_bind"].pop("effort")
  else:
    payload["capability_bind"]["api_key"] = "must-never-appear"

  with pytest.raises(ValueError, match="bind is invalid"):
    _verify(_resign(payload))


@pytest.mark.parametrize(
  ("verify_now_ns", "message"),
  [
    (_NOW_NS + 66_000_000_000, "expired"),
    (_NOW_NS - 6_000_000_000, "issued in the future"),
  ],
)
def test_envelope_rejects_expired_or_future_tokens(
  verify_now_ns: int,
  message: str,
) -> None:
  with pytest.raises(ValueError, match=message):
    _verify(_signed(), now_ns=verify_now_ns)


def test_envelope_rejects_excessive_ttl_bad_nonce_and_bad_channel() -> None:
  with pytest.raises(ValueError, match="ttl_seconds must be between"):
    _signed(ttl_seconds=301)
  with pytest.raises(ValueError, match="nonce must be 32 lowercase hex"):
    _signed(nonce="not-a-valid-nonce")
  with pytest.raises(ValueError, match="channel_id must be 64 lowercase"):
    _signed(channel_id="f" * 32)


@pytest.mark.parametrize(
  ("field", "value"),
  [
    ("task_id", "bg-other"),
    ("owner_user_id", "99"),
  ],
)
def test_ordinary_authority_rejects_resigned_identity_drift(
  field: str,
  value: str,
) -> None:
  payload = json.loads(_signed())
  payload[field] = value

  with pytest.raises(
    ValueError,
    match="ordinary autonomous session authority bindings",
  ):
    _verify(_resign(payload))


def test_envelope_rejects_non_autonomous_bind() -> None:
  with pytest.raises(ValueError, match="autonomous or cron run mode"):
    _signed(bind=_bind(run_mode="interactive"))


def test_envelope_retains_exact_credential_contract() -> None:
  user_bind = _bind(principal="user")
  user_authority = _ordinary_session_authority(bind=user_bind)
  envelope = _verify(
    _signed(
      bind=user_bind,
      session_authority=user_authority,
    )
  )
  assert envelope.bind.credential_ref == "autonomous-user:test-handle"
  assert (
    envelope.session_authority.to_gateway_session()
    .session_credential_handle
    is not None
  )

  mismatched = user_bind.model_copy(
    update={"credential_ref": "autonomous-user:different-handle"}
  )
  with pytest.raises(ValueError, match="credential authority does not match"):
    _signed(
      bind=mismatched,
      session_authority=user_authority,
    )


def test_superseded_contracts_and_in_memory_replay_path_are_deleted() -> None:
  source = Path(envelope_module.__file__).read_text(encoding="utf-8")

  assert "AUTONOMOUS_CAPABILITY_ENVELOPE_VERSION = 1" not in source
  assert "AUTONOMOUS_CAPABILITY_ENVELOPE_VERSION = 2" not in source
  assert "proof_authority" not in source
  assert "ProofAutonomous" not in source
  assert '"iat"' not in source
  assert '"exp"' not in source
  assert "used_nonces" not in source
  assert "MutableSet" not in source
  assert "legacy" not in source.lower()
