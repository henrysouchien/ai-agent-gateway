"""Validation entry point for mirrored commercial golden contract artifacts."""

from __future__ import annotations

import hashlib
from importlib.resources import files
import json
import math
from pathlib import Path
from decimal import Decimal
from datetime import datetime, timezone
from uuid import UUID

from jsonschema import Draft202012Validator

from .commercial_claims import (
  CommercialClaimError,
  CommercialClaimTrustSnapshot,
  CommercialClaimVerifier,
  VerifiedCommercialClaim,
)
from .commercial_work_authorization import (
  WorkAuthorizationError,
  WorkAuthorizationTrustSnapshot,
  WorkAuthorizationVerifier,
)


CONTRACT_FILES = frozenset({
  "commercial-claim.schema.json",
  "commercial-usage-event.schema.json",
  "commercial-usage-event-v2.schema.json",
  "commercial-claim-verification-v1.json",
  "commercial-work-authorization.schema.json",
  "commercial-work-authorization-verification-v1.json",
  "commercial-v1-fixtures.json",
  "canonicalization-v1-vectors.json",
})
MANIFEST_FILE = "commercial-contract-digests-v1.json"
USAGE_V3_CONTRACT_FILES = frozenset({"commercial-usage-event-v3.schema.json"})
USAGE_V3_MANIFEST_FILE = "commercial-usage-v3-digests.json"


def packaged_contract_directory() -> Path:
  return Path(str(files("agent_gateway") / "contracts" / "claim-contract-v1"))


def packaged_usage_v3_contract_directory() -> Path:
  return Path(str(files("agent_gateway") / "contracts" / "commercial-usage-v3"))


def verify_usage_v3_contract_directory(directory: Path) -> dict[str, str]:
  manifest = json.loads(
    (directory / USAGE_V3_MANIFEST_FILE).read_text(encoding="utf-8")
  )
  if manifest.get("contract_version") != "commercial-usage-v3":
    raise ValueError("commercial Usage V3 manifest version is unsupported")
  digests = manifest.get("digests")
  if not isinstance(digests, dict) or set(digests) != USAGE_V3_CONTRACT_FILES:
    raise ValueError("commercial Usage V3 manifest file set is invalid")
  for name in sorted(USAGE_V3_CONTRACT_FILES):
    data = (directory / name).read_bytes()
    actual = "sha256:" + hashlib.sha256(data).hexdigest()
    if digests[name] != actual:
      raise ValueError(f"commercial Usage V3 contract digest mismatch: {name}")
    Draft202012Validator.check_schema(json.loads(data))
  return dict(digests)


def verify_contract_directory(directory: Path) -> dict[str, str]:
  manifest = json.loads((directory / MANIFEST_FILE).read_text(encoding="utf-8"))
  if manifest.get("contract_version") != "commercial-cross-repository-contract-v1":
    raise ValueError("commercial contract manifest version is unsupported")
  digests = manifest.get("digests")
  if not isinstance(digests, dict) or set(digests) != CONTRACT_FILES:
    raise ValueError("commercial contract manifest file set is invalid")
  for name in sorted(CONTRACT_FILES):
    data = (directory / name).read_bytes()
    actual = "sha256:" + hashlib.sha256(data).hexdigest()
    if digests[name] != actual:
      raise ValueError(f"commercial contract digest mismatch: {name}")
  _verify_claim_cases(directory / "commercial-claim-verification-v1.json")
  _verify_work_authorization_cases(
    directory / "commercial-work-authorization-verification-v1.json"
  )
  _verify_usage_cases(
    directory / "commercial-v1-fixtures.json",
    directory / "commercial-usage-event.schema.json",
    directory / "commercial-usage-event-v2.schema.json",
  )
  _verify_canonicalization_vectors(directory / "canonicalization-v1-vectors.json")
  return dict(digests)


def _verify_claim_cases(path: Path) -> None:
  fixture = json.loads(path.read_text(encoding="utf-8"))
  trust = fixture["trust"]
  agreement_id = UUID(trust["agreement_id"])
  verifier = CommercialClaimVerifier(CommercialClaimTrustSnapshot(
    public_keys_by_id={
      key_id: pem.encode("ascii")
      for key_id, pem in fixture["public_keys_by_id"].items()
    },
    environment=trust["environment"],
    manifest_versions=frozenset(trust["manifest_versions"]),
    payer_policy_versions=frozenset(trust["payer_policy_versions"]),
    budget_policy_versions=frozenset(trust["budget_policy_versions"]),
    shadow_rate_versions=frozenset(trust["shadow_rate_versions"]),
    current_agreement_terms_revision=lambda candidate: (
      trust["current_agreement_terms_revision"]
      if candidate == agreement_id else None
    ),
  ))
  observed = []
  for case in fixture["cases"]:
    try:
      verifier.verify_for_work_start(
        case["token"], now=fixture["verification_time"]
      )
      observed.append("accepted")
    except CommercialClaimError as exc:
      observed.append(exc.code)
  expected = [case["expected"] for case in fixture["cases"]]
  if observed != expected:
    raise ValueError(
      f"commercial claim fixture outcomes differ: expected={expected!r} observed={observed!r}"
    )


def _verify_work_authorization_cases(path: Path) -> None:
  fixture = json.loads(path.read_text(encoding="utf-8"))
  trust = fixture["trust"]
  context_id = UUID(trust["execution_claim_jti"])
  execution_claim = VerifiedCommercialClaim(
    schema_version=1,
    key_id="commercial-signing-fixture-v1",
    subject="user:123",
    environment=trust["environment"],
    surface="hp1",
    commercial_account_id=UUID("11111111-1111-4111-8111-111111111111"),
    agreement_id=UUID("22222222-2222-4222-8222-222222222222"),
    agreement_terms_revision=2,
    offer_code="hp1_pro",
    effective_scopes=tuple(trust["effective_scopes"]),
    entitlement_revision=42,
    payer_policy_version="hp1_customer_host@v1",
    budget_policy_version="hp1_pro_budget@v1",
    shadow_rate_version="commercial_rates@2026-09-01",
    manifest_version=trust["manifest_version"],
    authorized_work_start_deadline=fixture["verification_time"] + 300,
    usage_accept_until=fixture["verification_time"] + 3600,
    issued_at=fixture["verification_time"] - 1,
    expires_at=fixture["verification_time"] + 300,
    context_id=context_id,
  )
  observed = []
  for case in fixture["cases"]:
    verification_time = case.get(
      "verification_time", fixture["verification_time"]
    )
    verifier = WorkAuthorizationVerifier(
      WorkAuthorizationTrustSnapshot(
        public_keys_by_id={
          key_id: pem.encode("ascii")
          for key_id, pem in fixture["public_keys_by_id"].items()
        },
        environment=trust["environment"],
        manifest_version=trust["manifest_version"],
        capability_required_scopes={
          capability: frozenset(scopes)
          for capability, scopes in trust[
            "capability_required_scopes"
          ].items()
        },
        capability_allowed_operations={
          capability: frozenset(operations)
          for capability, operations in trust[
            "capability_allowed_operations"
          ].items()
        },
        retired_at_by_key_id=fixture["retired_at_by_key_id"],
      ),
      clock=lambda verification_time=verification_time: verification_time,
    )
    try:
      verifier.verify_for_attach(
        case["token"],
        execution_claim=execution_claim,
        request_id=case.get("request_id", trust["request_id"]),
        session_id=case.get("session_id", trust["session_id"]),
        operation=case.get("operation", trust["operation"]),
        provider=case.get("provider", trust["provider"]),
        billing_mode=case.get("billing_mode", trust["billing_mode"]),
        capability_id=case.get(
          "capability_id", next(iter(trust["capability_required_scopes"]))
        ),
      )
      observed.append("accepted")
    except WorkAuthorizationError as exc:
      observed.append(exc.code)
  expected = [case["expected"] for case in fixture["cases"]]
  if observed != expected:
    raise ValueError(
      "commercial work-authorization fixture outcomes differ: "
      f"expected={expected!r} observed={observed!r}"
    )


def _verify_usage_cases(
  path: Path,
  schema_path: Path,
  schema_v2_path: Path,
) -> None:
  fixture = json.loads(path.read_text(encoding="utf-8"))
  validator = Draft202012Validator(json.loads(schema_path.read_text(encoding="utf-8")))
  validator_v2 = Draft202012Validator(
    json.loads(schema_v2_path.read_text(encoding="utf-8"))
  )
  usage = fixture.get("usage")
  if not isinstance(usage, dict) or not usage:
    raise ValueError("commercial usage fixtures are absent")
  for name, event in usage.items():
    _validate_usage_event(event, name=f"usage.{name}", validator=validator)
  usage_v2 = fixture.get("usage_v2")
  if not isinstance(usage_v2, dict) or not usage_v2:
    raise ValueError("commercial Usage V2 fixtures are absent")
  for name, event in usage_v2.items():
    _validate_usage_event(
      event, name=f"usage_v2.{name}", validator=validator_v2
    )
  submissions = fixture.get("idempotency", {}).get("submissions")
  if not isinstance(submissions, list) or not submissions:
    raise ValueError("commercial idempotency usage fixtures are absent")
  for index, event in enumerate(submissions):
    _validate_usage_event(
      event, name=f"idempotency.submissions[{index}]", validator=validator
    )


def _validate_usage_event(event: object, *, name: str, validator) -> None:
  if not isinstance(event, dict):
    raise ValueError(f"commercial usage fixture is not an object: {name}")
  errors = sorted(validator.iter_errors(event), key=lambda error: list(error.path))
  if errors:
    raise ValueError(f"commercial usage fixture schema mismatch: {name}: {errors[0].message}")
  reasoning = event.get("reasoning_tokens_observed")
  if reasoning is not None and reasoning > event["billable_output_tokens"]:
    raise ValueError(f"commercial usage fixture double-counts reasoning: {name}")
  digest = event["source_payload_sha256"]
  actual = canonical_usage_payload_sha256(event)
  if digest != actual:
    raise ValueError(f"commercial usage fixture canonical digest mismatch: {name}")


def _normalized_usage_body(event: dict) -> dict:
  body = dict(event)
  body.pop("source_payload_sha256", None)
  for field in (
    "separately_billed_tool_cost_usd", "producer_estimated_cost_usd",
    "provider_reported_cost_usd",
  ):
    if body.get(field) is not None:
      body[field] = Decimal(str(body[field]))
  return body


def canonical_usage_payload_sha256(event: dict) -> str:
  body = _normalized_usage_body(event)
  return "sha256:" + hashlib.sha256(_canonical_json(body).encode("utf-8")).hexdigest()


def _verify_canonicalization_vectors(path: Path) -> None:
  fixture = json.loads(path.read_text(encoding="utf-8"))
  vectors = fixture.get("vectors")
  if not isinstance(vectors, list) or not vectors:
    raise ValueError("commercial canonicalization vectors are absent")
  for vector in vectors:
    value = json.loads(vector["input_json"])
    for pointer, field_type in vector.get("field_types", {}).items():
      key = pointer.removeprefix("/")
      if "/" in key or key not in value:
        raise ValueError(f"unsupported commercial vector pointer: {pointer}")
      if field_type == "decimal":
        value[key] = Decimal(value[key])
      elif field_type == "datetime":
        value[key] = datetime.fromisoformat(value[key].replace("Z", "+00:00"))
      elif field_type == "uuid":
        value[key] = UUID(value[key])
      else:
        raise ValueError(f"unsupported commercial vector field type: {field_type}")
    canonical = _canonical_json(value)
    digest = "sha256:" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    if canonical != vector["canonical_utf8"] or digest != vector["sha256"]:
      raise ValueError(f"commercial canonicalization vector mismatch: {vector['name']}")
  usage = fixture["usage_event_digest_vector"]
  normalized = usage["normalized_payload"]
  canonical = _canonical_json(_normalized_usage_body(normalized))
  digest = "sha256:" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()
  if canonical != usage["canonical_utf8"] or digest != usage["sha256"]:
    raise ValueError("commercial usage canonicalization vector mismatch")


def _canonical_json(value) -> str:
  if value is None:
    return "null"
  if value is True:
    return "true"
  if value is False:
    return "false"
  if type(value) is int:
    return str(value)
  if type(value) is float:
    if not math.isfinite(value):
      raise ValueError("commercial canonical float must be finite")
    return _canonical_json(Decimal(str(value)))
  if isinstance(value, Decimal):
    if not value.is_finite():
      raise ValueError("commercial canonical decimal must be finite")
    rendered = format(value, "f")
    if "." in rendered:
      rendered = rendered.rstrip("0").rstrip(".")
    return "0" if rendered in {"-0", ""} else rendered
  if isinstance(value, str):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
  if isinstance(value, list):
    return "[" + ",".join(_canonical_json(item) for item in value) + "]"
  if isinstance(value, dict):
    return "{" + ",".join(
      _canonical_json(key) + ":" + _canonical_json(value[key])
      for key in sorted(value)
    ) + "}"
  if isinstance(value, datetime):
    if value.tzinfo is None or value.utcoffset() is None:
      raise ValueError("commercial canonical datetime must be timezone-aware")
    rendered = value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
    return _canonical_json(rendered)
  if isinstance(value, UUID):
    return _canonical_json(str(value))
  raise ValueError(f"unsupported commercial canonical type: {type(value).__name__}")
