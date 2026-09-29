from __future__ import annotations

from gateway_test_support.autonomous_launch_test_support import (
  _control_authority,
  _memory_control_authority,
  _workload,
  _bind,
  _credential_handle,
  _ordinary_session_authority,
)

from collections.abc import Callable
import hashlib
import hmac
import json
from pathlib import Path
from typing import Literal, TypedDict

import pytest

import agent_gateway.autonomous_launch_envelope as envelope_module
from agent_gateway.autonomous_launch_envelope import (
  AUTONOMOUS_CAPABILITY_ENVELOPE_AUDIENCE,
  AUTONOMOUS_CAPABILITY_ENVELOPE_VERSION,
  AUTONOMOUS_RUNTIME_SESSION_PURPOSE,
  AutonomousControlAuthority,
  AutonomousDispatchScope,
  AutonomousLaunchEnvelope,
  AutonomousLaunchWorkload,
  AutonomousSessionAuthority,
  OrdinaryAutonomousSessionAuthority,
  sign_autonomous_launch_envelope,
  verify_autonomous_launch_envelope,
)
from model_authority.bind import CapabilityBind
from model_authority.binding import CredentialHandle, CredentialPrincipal, RunMode
from agent_gateway.session import GatewaySession
from agent_gateway.skill_limits import SkillExecutionLimits
from agent_gateway.agent_session_log_layout import (
  AutonomousSessionLogAuthority,
)


_SECRET = "launch-envelope-test-secret-at-least-32-bytes"
_NOW_NS = 1_800_000_000_000_000_000
_NONCE = "0123456789abcdef0123456789abcdef"
_CHANNEL_ID = "12" * 32

class _DefaultOverride:
  pass


class _WorkloadOverrides(TypedDict, total=False):
  profile: str
  mode: Literal["run_once", "task", "skill", "pack"]
  task: str | None
  skill: str | None
  pack: str | None
  context: str | None
  ticker: str | None
  dev_mode: bool
  max_budget_usd: float | None
  deliver: bool
  admitted_skill_execution_limits: SkillExecutionLimits | None


class _ControlAuthorityOverrides(TypedDict, total=False):
  admission_ledger_path: str
  operator_inbox_path: str


_DEFAULT_OVERRIDE = _DefaultOverride()















def test_ordinary_authority_accepts_invite_role_exactly() -> None:
  session = _ordinary_session_authority(role="invite").to_gateway_session()
  assert session.role == "invite"


@pytest.mark.parametrize("role", ["Owner", " owner ", "OWNER", "", None, True])
def test_ordinary_authority_rejects_malformed_role(role: object) -> None:
  receipt = _ordinary_session_authority().receipt()
  receipt["ordinary_authority"]["role"] = role
  with pytest.raises(ValueError, match="role must be exactly"):
    AutonomousSessionAuthority.from_receipt(receipt)


def _signed(
  *,
  task_id: str = "bg_7",
  control_run_id: str = "run-7",
  owner_user_id: str = "42",
  channel_id: str = _CHANNEL_ID,
  bind: CapabilityBind | _DefaultOverride = _DEFAULT_OVERRIDE,
  workload: AutonomousLaunchWorkload | _DefaultOverride = _DEFAULT_OVERRIDE,
  control_authority: AutonomousControlAuthority | _DefaultOverride = (
    _DEFAULT_OVERRIDE
  ),
  session_authority: AutonomousSessionAuthority | _DefaultOverride = (
    _DEFAULT_OVERRIDE
  ),
  ttl_seconds: int = 60,
  now_ns: int | None = _NOW_NS,
  nonce: str | None = _NONCE,
) -> str:
  resolved_bind = _bind() if isinstance(bind, _DefaultOverride) else bind
  resolved_workload = (
    _workload() if isinstance(workload, _DefaultOverride) else workload
  )
  resolved_control_authority = (
    _control_authority()
    if isinstance(control_authority, _DefaultOverride)
    else control_authority
  )
  resolved_session_authority = (
    _ordinary_session_authority(bind=resolved_bind)
    if isinstance(session_authority, _DefaultOverride)
    else session_authority
  )
  return sign_autonomous_launch_envelope(
    _SECRET,
    task_id=task_id,
    control_run_id=control_run_id,
    owner_user_id=owner_user_id,
    channel_id=channel_id,
    bind=resolved_bind,
    workload=resolved_workload,
    control_authority=resolved_control_authority,
    session_authority=resolved_session_authority,
    ttl_seconds=ttl_seconds,
    now_ns=now_ns,
    nonce=nonce,
  )


def _verify(
  envelope_json: str,
  *,
  now_ns: int | None = _NOW_NS,
) -> AutonomousLaunchEnvelope:
  return verify_autonomous_launch_envelope(
    _SECRET,
    envelope_json,
    now_ns=now_ns,
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


def test_v6_round_trip_constructs_exact_gateway_session() -> None:
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
  assert envelope.version == AUTONOMOUS_CAPABILITY_ENVELOPE_VERSION == 6
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


def test_v6_skill_limits_are_signed_required_and_compare_false() -> None:
  first = _workload(
    mode="skill",
    skill="quant-research",
    admitted_skill_execution_limits=SkillExecutionLimits(
      20,
      32_000,
      20.0,
    ),
  )
  second = _workload(
    mode="skill",
    skill="quant-research",
    admitted_skill_execution_limits=SkillExecutionLimits(
      1,
      2,
      3.0,
    ),
  )

  assert first == second
  assert first.receipt()["admitted_skill_execution_limits"] == {
    "max_turns": 20,
    "max_tokens": 32_000,
    "max_budget_usd": 20.0,
  }
  parsed = AutonomousLaunchWorkload.from_receipt(first.receipt())
  assert type(parsed.admitted_skill_execution_limits) is SkillExecutionLimits
  assert parsed.admitted_skill_execution_limits == (
    SkillExecutionLimits(20, 32_000, 20.0)
  )


@pytest.mark.parametrize(
  "value",
  [
    None,
    {},
    {"max_turns": 20, "max_tokens": 32_000},
    {
      "max_turns": 20,
      "max_tokens": 32_000,
      "max_budget_usd": 20.0,
      "extra": None,
    },
    {"max_turns": True, "max_tokens": 32_000, "max_budget_usd": 20.0},
  ],
)
def test_v6_skill_workload_rejects_malformed_signed_limits(value: object) -> None:
  payload = json.loads(_signed(
    workload=_workload(mode="skill", skill="quant-research")
  ))
  payload["workload"]["admitted_skill_execution_limits"] = value

  with pytest.raises(ValueError, match="workload is invalid"):
    _verify(_resign(payload))


def test_v6_non_skill_workload_rejects_signed_limits_object() -> None:
  payload = json.loads(_signed())
  payload["workload"]["admitted_skill_execution_limits"] = {
    "max_turns": None,
    "max_tokens": None,
    "max_budget_usd": None,
  }

  with pytest.raises(ValueError, match="workload is invalid"):
    _verify(_resign(payload))


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


def test_v6_sign_rejects_workload_without_session_log_authority() -> None:
  workload = AutonomousLaunchWorkload(
    profile="analyst",
    mode="run_once",
    task=None,
    skill=None,
    pack=None,
    context=None,
    ticker=None,
    research_file_id=None,
    dev_mode=False,
    max_budget_usd=None,
    deliver=True,
    admitted_skill_execution_limits=None,
  )
  with pytest.raises(TypeError, match="requires exact session-log authority"):
    sign_autonomous_launch_envelope(
      _SECRET,
      task_id="bg_7",
      control_run_id="run-7",
      owner_user_id="42",
      channel_id=_CHANNEL_ID,
      bind=_bind(),
      workload=workload,
      control_authority=_control_authority(),
      session_authority=_ordinary_session_authority(),
      now_ns=_NOW_NS,
      nonce=_NONCE,
    )


def test_v6_verify_rejects_missing_session_log_authority() -> None:
  payload = json.loads(_signed())
  payload["workload"].pop("session_log_authority")
  with pytest.raises(ValueError, match="workload is invalid"):
    _verify(_resign(payload))


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
  mutation: Callable[[dict[str, object]], object],
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
  kwargs: _WorkloadOverrides,
) -> None:
  with pytest.raises(ValueError, match="autonomous launch workload"):
    _workload(**kwargs)


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
    {"admission_ledger_path": "/tmp/../tmp/autonomous-admissions.sqlite3"},
  ],
)
def test_control_authority_rejects_noncanonical_or_aliased_paths(
  kwargs: _ControlAuthorityOverrides,
) -> None:
  with pytest.raises(ValueError, match="autonomous control authority"):
    _control_authority(**kwargs)


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
