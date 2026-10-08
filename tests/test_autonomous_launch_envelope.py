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
from agent_workflow_contracts import ResolvedAuthority
from agent_workflow_contracts.models import (
  AdmittedContextWire, AdmittedLineage, AdmittedResearchBinding, AdmittedTenant, ToolGrant,
)

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
  admitted_context: AdmittedContextWire | None = None,
  tool_authority: ResolvedAuthority | None = None,
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
    admitted_context=admitted_context,
    tool_authority=tool_authority,
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


def test_v7_round_trip_constructs_exact_gateway_session() -> None:
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
  assert envelope.version == AUTONOMOUS_CAPABILITY_ENVELOPE_VERSION == 7
  assert envelope.task_id == "bg_7"
  assert envelope.control_run_id == "run-7"
  assert envelope.owner_user_id == "42"
  assert envelope.channel_id == _CHANNEL_ID
  assert envelope.bind == _bind()
  assert envelope.workload == _workload()
  assert envelope.control_authority == _control_authority()
  assert envelope.session_authority == authority
  assert envelope.admitted is None
  assert parsed["admitted_context"] is parsed["tool_authority"] is None

  session = envelope.session_authority.to_gateway_session()
  assert type(session) is GatewaySession
  assert session.session_id == "bg_7"
  assert session.user_id == "42"
  assert session.owner_user_id == "42"
  assert session.purpose == AUTONOMOUS_RUNTIME_SESSION_PURPOSE
  assert session.dispatch_scope == dispatch_scope.receipt()


def test_skill_limit_observations_are_signed_but_not_cli_arguments() -> None:
  first = _workload(
    mode="skill",
    skill="quant-research",
    admitted_skill_execution_limits=SkillExecutionLimits(32_000),
  )
  second = _workload(
    mode="skill",
    skill="quant-research",
    admitted_skill_execution_limits=SkillExecutionLimits(2),
  )

  assert first == second
  assert first.receipt()["admitted_skill_execution_limits"] == {
    "max_tokens": 32_000,
  }
  parsed = _verify(_signed(workload=first)).workload
  assert type(parsed.admitted_skill_execution_limits) is SkillExecutionLimits
  assert parsed.admitted_skill_execution_limits == SkillExecutionLimits(32_000)

  payload = json.loads(_signed(workload=first))
  payload["workload"]["admitted_skill_execution_limits"] = {"max_tokens": 2}
  with pytest.raises(ValueError, match="signature is invalid"):
    _verify(json.dumps(payload, sort_keys=True, separators=(",", ":")))


@pytest.mark.parametrize(
  "value",
  [
    {},
    {"max_turns": 20, "max_tokens": 32_000},
    {
      "max_turns": 20,
      "max_tokens": 32_000,
      "max_budget_usd": 20.0,
      "extra": None,
    },
    {"max_tokens": True},
    {"max_tokens": 0},
  ],
)
def test_v7_skill_workload_rejects_malformed_signed_limits(value: object) -> None:
  payload = json.loads(_signed(
    workload=_workload(mode="skill", skill="quant-research")
  ))
  payload["workload"]["admitted_skill_execution_limits"] = value

  with pytest.raises(ValueError, match="workload is invalid"):
    _verify(_resign(payload))


@pytest.mark.parametrize("observation", ["absent", "null"])
def test_signed_skill_workload_without_limit_observation(
  observation: str,
) -> None:
  workload = AutonomousLaunchWorkload(
    profile="analyst",
    mode="skill",
    task=None,
    skill="quant-research",
    pack=None,
    context=None,
    ticker=None,
    research_file_id=None,
    dev_mode=False,
    deliver=True,
    session_log_authority=_workload().session_log_authority,
  )
  raw = _signed(workload=workload)
  if observation == "absent":
    payload = json.loads(raw)
    payload["workload"].pop("admitted_skill_execution_limits")
    raw = _resign(payload)

  verified = _verify(raw)
  assert verified.workload.skill == "quant-research"
  assert verified.workload.admitted_skill_execution_limits is None
  assert verified.workload.receipt()["admitted_skill_execution_limits"] is None


@pytest.mark.parametrize(
  "kwargs",
  [
    {"mode": "run_once"},
    {"mode": "task", "task": "Investigate the variance"},
    {"mode": "pack", "pack": "daily-risk-pack"},
  ],
)
def test_signed_non_skill_workload_carries_limit_observation(
  kwargs: _WorkloadOverrides,
) -> None:
  overrides: _WorkloadOverrides = {
    **kwargs,
    "admitted_skill_execution_limits": SkillExecutionLimits(32_000),
  }
  workload = _workload(**overrides)
  verified = _verify(_signed(workload=workload))

  assert verified.workload.mode == workload.mode
  assert verified.workload.admitted_skill_execution_limits == SkillExecutionLimits(
    32_000
  )
  assert verified.payload()["workload"]["admitted_skill_execution_limits"] == {
    "max_tokens": 32_000,
  }


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


def test_v7_sign_rejects_workload_without_session_log_authority() -> None:
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


def test_v7_verify_rejects_missing_session_log_authority() -> None:
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
    {"mode": "skill", "skill": None},
    {"mode": "skill", "skill": "review", "task": "other"},
  ],
)
def test_workload_rejects_incompatible_or_noncanonical_fields(
  kwargs: _WorkloadOverrides,
) -> None:
  with pytest.raises(ValueError, match="autonomous launch workload"):
    _workload(**kwargs)


@pytest.mark.parametrize(
  "kwargs",
  [
    {"mode": "task", "task": "\U0001f642" * (64 * 1024)},
    {
      "mode": "skill",
      "skill": "review",
      "context": "\U0001f642" * (64 * 1024),
    },
    {"mode": "skill", "skill": "s" * 513},
    {"mode": "skill", "skill": "review", "ticker": "t" * 513},
    {"mode": "pack", "pack": "p" * 513},
  ],
)
def test_signed_workload_preserves_text_above_former_byte_ceilings(
  kwargs: _WorkloadOverrides,
) -> None:
  workload = _workload(**kwargs)
  verified = _verify(_signed(workload=workload))

  assert verified.workload == workload
  assert verified.payload()["workload"] == workload.receipt()


def test_workload_accepts_profile_above_former_byte_ceiling() -> None:
  profile = "p" * 513
  workload = _workload(profile=profile)
  assert workload.profile == profile


@pytest.mark.parametrize(
  "kwargs",
  [
    {"profile": "ana\x00lyst"},
    {"mode": "task", "task": "review\x00task"},
    {"mode": "skill", "skill": "re\x00view"},
    {"mode": "skill", "skill": "review", "context": "file\x00context"},
    {"mode": "skill", "skill": "review", "ticker": "MS\x00FT"},
    {"mode": "pack", "pack": "risk\x00pack"},
  ],
)
def test_workload_rejects_nul_before_process_arguments(
  kwargs: _WorkloadOverrides,
) -> None:
  with pytest.raises(ValueError, match="autonomous launch workload"):
    _workload(**kwargs)


@pytest.mark.parametrize(
  ("kwargs", "field_name"),
  [
    ({"mode": "run_once"}, "profile"),
    ({"mode": "task", "task": "review"}, "task"),
    ({"mode": "skill", "skill": "review"}, "skill"),
    ({"mode": "skill", "skill": "review"}, "context"),
    ({"mode": "skill", "skill": "review"}, "ticker"),
    ({"mode": "pack", "pack": "risk-pack"}, "pack"),
  ],
)
def test_signed_workload_rejects_nul_before_process_arguments(
  kwargs: _WorkloadOverrides,
  field_name: str,
) -> None:
  payload = json.loads(_signed(workload=_workload(**kwargs)))
  payload["workload"][field_name] = "invalid\x00argument"

  with pytest.raises(ValueError, match="workload is invalid"):
    _verify(_resign(payload))


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


def test_envelope_carries_a_long_ttl_and_expiry_bounds_it() -> None:
  envelope = _signed(ttl_seconds=600)

  _verify(envelope, now_ns=_NOW_NS + 590_000_000_000)
  with pytest.raises(ValueError, match="expired"):
    _verify(envelope, now_ns=_NOW_NS + 606_000_000_000)
  with pytest.raises(ValueError, match="ttl_seconds must be positive"):
    _signed(ttl_seconds=0)


def test_envelope_rejects_bad_nonce_and_bad_channel() -> None:
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


def _admitted_authority(bind: CapabilityBind) -> tuple[AdmittedContextWire, ResolvedAuthority]:
  return (
    AdmittedContextWire(
      tenant=AdmittedTenant(tenant_id="test-product", storage_id="42", role="owner"),
      research_binding=AdmittedResearchBinding(research_file_id=7, ticker="MSFT"),
      capability_bind=bind,
      children=(),
      tool_grant=(),
      budget_share=None,
      env_projection=("PATH",),
      approval_route="parent_delegated",
      handle_grants=(),
      placement="local_stub",
      lineage=AdmittedLineage(parent_job_id="parent-7", depth=1),
    ),
    ResolvedAuthority(
      operation_name="risk.scan",
      grant=ToolGrant(grant_id="grant-7", tools=(), digest="sha256:" + "0" * 64),
    ),
  )


def test_v7_round_trip_carries_exact_admitted_authority() -> None:
  bind = _bind().model_copy(update={"capability_id": "node.explore"})
  context, authority = _admitted_authority(bind)
  raw = _signed(bind=bind, admitted_context=context, tool_authority=authority)
  verified = _verify(raw)
  broker = envelope_module._decode_broker_verified_autonomous_launch_envelope(raw, now_ns=_NOW_NS)
  assert verified.admitted == broker.admitted == (context, authority)
  assert verified.bind == context.capability_bind
  assert verified.payload() == json.loads(raw)


@pytest.mark.parametrize("field", ["admitted_context", "tool_authority"])
def test_envelope_signature_binds_admitted_authority(field: str) -> None:
  context, authority = _admitted_authority(_bind())
  payload = json.loads(_signed(admitted_context=context, tool_authority=authority))
  payload[field] = None
  with pytest.raises(ValueError, match="signature is invalid"):
    _verify(json.dumps(payload, sort_keys=True, separators=(",", ":")))


def test_envelope_rejects_mismatched_admitted_bind() -> None:
  context, authority = _admitted_authority(_bind())
  changed = context.model_copy(update={"capability_bind": _bind().model_copy(update={"credential_ref": "service:different-handle"})})
  with pytest.raises(ValueError, match="bind does not match admitted context"):
    _signed(admitted_context=changed, tool_authority=authority)
  payload = json.loads(_signed(admitted_context=context, tool_authority=authority))
  payload["admitted_context"] = changed.model_dump(mode="json")
  with pytest.raises(ValueError, match="bind does not match admitted context"):
    _verify(_resign(payload))


@pytest.mark.parametrize("field", ["admitted_context", "tool_authority"])
@pytest.mark.parametrize("mutation", ["null", "unknown", "nul", "noncanonical"])
def test_envelope_admitted_authority_keeps_closed_canonical_contract(field: str, mutation: str) -> None:
  context, authority = _admitted_authority(_bind())
  payload = json.loads(_signed(admitted_context=context, tool_authority=authority))
  if mutation == "null":
    payload[field] = None
  elif mutation == "unknown":
    payload[field]["unknown"] = "untrusted"
  elif mutation == "nul":
    if field == "tool_authority":
      payload[field]["operation_name"] = "risk\x00scan"
    else:
      payload[field]["tenant"]["storage_id"] = "42\x00"
  elif field == "tool_authority":
    payload[field]["grant"]["grant_id"] = " grant-7 "
  else:
    payload[field]["research_binding"]["ticker"] = " MSFT "
  with pytest.raises(ValueError):
    _verify(_resign(payload))
