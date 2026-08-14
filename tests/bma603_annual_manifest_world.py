"""Small schema-only annual-manifest contract fixture.

The former fixture rebuilt a historical canonical model-state world and crossed
retired persistence authorities.  This replacement keeps only the useful
contract seam: a strict ``BuildManifest`` and an independently validated
``AnnualManifestInputClosure``.  It has no storage, runtime, or predecessor
dependency.
"""

from __future__ import annotations

from dataclasses import dataclass

from schema.annual_acceptance_foundation import (
    AnnualDataLineageBinding,
    annual_data_lineage_digest,
)
from schema.annual_manifest_slot_registry import ANNUAL_MANIFEST_INPUT_SLOTS
from schema.build_manifest import (
    ANNUAL_MANIFEST_PRODUCER_CONTRACT_ID,
    AnnualManifestInputClosure,
    AnnualManifestInputClosureEntry,
    BuildManifest,
    DirectCanonicalBuildManifestAncestry,
    annual_manifest_input_closure_content_sha256,
)
from schema.model_state_identity import ModelStateRef, format_model_state_uri
from schema.stable_control_identity import FiscalAxis, FiscalPeriod
from schema.versioned_artifact import (
    CanonicalResearchFileArtifactScope,
    GlobalArtifactScope,
    VersionedArtifactRef,
    compute_artifact_content_sha256,
    format_versioned_artifact_uri,
)
from schema.workspace_producer_authority import (
    WorkspaceProducerRegistrationToken,
    registration_token_digest,
)


_HASH_A = "a" * 64
_HASH_B = "b" * 64
_USER_ID = "bma603-contract-user"
_RESEARCH_FILE_ID = 603
_TICKER = "PCTY"

_REQUIRED_ARTIFACT_KINDS = {
    "business_model": "business_model",
    "model_build_context": "model_build_context",
    "upstream_artifact_bundle": "annual_acceptance_input_artifact_bundle",
    "driver_assumption_plan": "driver_assumption_plan",
    "stable_control_identity_registry": "stable_control_identity_registry",
    "archetype_required_capability_profile": (
        "archetype_required_capability_profile"
    ),
    "acceptance_policy": "acceptance_policy",
    "requirement_catalog": "requirement_catalog",
    "requirement_policy_catalog": "requirement_policy_catalog",
    "requirement_policy_evaluator_registry": (
        "requirement_policy_evaluator_registry"
    ),
    "selected_requirement_set": "selected_requirement_set",
    "selected_annual_requirement_gap_projection": (
        "selected_annual_requirement_gap_projection"
    ),
    "annual_data_requirement_baseline_matrix": (
        "annual_data_requirement_baseline_matrix"
    ),
    "valuation_peer_universe_policy": "valuation_peer_universe_policy",
    "issuer_universe_snapshot": "issuer_universe_snapshot",
    "valuation_peer_universe_snapshot": "valuation_peer_universe_snapshot",
    "annual_baseline_fulfillment_manifest": (
        "annual_baseline_fulfillment_manifest"
    ),
    "historical_window_policy": "historical_window_policy",
    "statement_observation_snapshot": "statement_observation_snapshot",
    "source_routing_policy_snapshot": "source_routing_policy_snapshot",
    "source_arbitration_closure": "source_arbitration_closure",
    "statement_coverage_manifest": "statement_coverage_manifest",
    "operating_coverage_manifest": "operating_coverage_manifest",
    "adjustment_coverage_manifest": "adjustment_coverage_manifest",
    "forecast_coverage_manifest": "forecast_coverage_manifest",
    "valuation_comps_coverage_manifest": "valuation_comps_coverage_manifest",
    "runtime_identity_snapshot": "runtime_identity_snapshot",
    "forecast_control_snapshot": "forecast_control_snapshot",
    "thesis": "thesis",
    "handoff": "handoff",
}

_GLOBAL_PREFIXES = {
    "acceptance_policy",
    "requirement_catalog",
    "requirement_policy_catalog",
    "requirement_policy_evaluator_registry",
    "valuation_peer_universe_policy",
    "issuer_universe_snapshot",
    "historical_window_policy",
    "runtime_identity_snapshot",
}

_ANNUAL_DATA_PREFIXES = (
    "requirement_catalog",
    "requirement_policy_catalog",
    "requirement_policy_evaluator_registry",
    "selected_requirement_set",
    "annual_data_requirement_baseline_matrix",
    "valuation_peer_universe_policy",
    "issuer_universe_snapshot",
    "valuation_peer_universe_snapshot",
    "annual_baseline_fulfillment_manifest",
    "historical_window_policy",
    "source_routing_policy_snapshot",
    "source_arbitration_closure",
)


@dataclass(frozen=True, slots=True)
class Bma603AnnualManifestContractWorld:
    manifest: BuildManifest
    closure: AnnualManifestInputClosure


def _state_ref() -> ModelStateRef:
    return ModelStateRef(
        uri=format_model_state_uri(
            scope="canonical",
            user_id=_USER_ID,
            research_file_id=_RESEARCH_FILE_ID,
            ticker=_TICKER,
            revision=1,
        ),
        scope="canonical",
        user_id=_USER_ID,
        research_file_id=_RESEARCH_FILE_ID,
        ticker=_TICKER,
        revision=1,
        content_hash=_HASH_B,
    )


def _artifact_ref(
    artifact_kind: str,
    *,
    global_scope: bool = False,
) -> VersionedArtifactRef:
    scope = (
        GlobalArtifactScope()
        if global_scope
        else CanonicalResearchFileArtifactScope(
            user_id=_USER_ID,
            research_file_id=_RESEARCH_FILE_ID,
            ticker=_TICKER,
        )
    )
    artifact_id = f"{artifact_kind}-contract"
    payload = {
        "schema_version": f"{artifact_kind.replace('_', '-')}.v1",
        "artifact_id": artifact_id,
    }
    return VersionedArtifactRef(
        ref_schema_version="versioned-artifact-ref.v1",
        provider_id=(
            "artifact_store.global.v1"
            if global_scope
            else "artifact_store.canonical_research_file.v1"
        ),
        artifact_kind=artifact_kind,
        artifact_id=artifact_id,
        artifact_version=1,
        scope=scope,
        uri=format_versioned_artifact_uri(
            scope=scope,
            artifact_kind=artifact_kind,
            artifact_id=artifact_id,
            artifact_version=1,
        ),
        payload_schema_version=payload["schema_version"],
        media_type="application/json",
        digest_codec="canonical_json_v1",
        content_sha256=compute_artifact_content_sha256(
            payload,
            "canonical_json_v1",
        ),
    )


def _manifest() -> BuildManifest:
    state_ref = _state_ref()
    values: dict[str, object] = {}
    for prefix, artifact_kind in _REQUIRED_ARTIFACT_KINDS.items():
        ref = _artifact_ref(
            artifact_kind,
            global_scope=prefix in _GLOBAL_PREFIXES,
        )
        values[f"{prefix}_ref"] = ref
        values[f"{prefix}_hash"] = ref.content_sha256

    values.update(
        {
            "schema_version": "build-manifest.v2",
            "manifest_id": "bma603-contract-manifest",
            "manifest_version": 1,
            "requested_acceptance_class": "draft",
            "acceptance_eligibility": "non_annual_full",
            "manifest_emission_id": None,
            "emission_transaction_id": None,
            "annual_manifest_input_closure_ref": None,
            "annual_manifest_input_closure_hash": None,
            "annual_manifest_slot_registry_ref": None,
            "annual_manifest_slot_registry_hash": None,
            "producer_registration_token": None,
            "producer_runtime_identity_ref": None,
            "producer_runtime_identity_hash": None,
            "model_state_ref": state_ref,
            "annual_acceptance_lineage": None,
            "annual_data_lineage": None,
            "candidate_store_uri": "model-state://candidate/bma603-contract",
            "authoritative_store_uri": "model-state://canonical/bma603-contract",
            "candidate_store_hash": _HASH_A,
            "authoritative_base_hash": _HASH_B,
            "annual_data_requirement_baseline_matrix_id": "bma603-contract",
            "annual_data_requirement_baseline_matrix_version": 1,
            "annual_data_requirement_baseline_entries_digest": _HASH_A,
            "acceptance_evidence_cutoff_policy_ref": None,
            "acceptance_evidence_cutoff_policy_hash": None,
            "acceptance_evidence_cutoff_binding_ref": None,
            "acceptance_evidence_cutoff_binding_hash": None,
            "acceptance_evidence_catalog_seal_ref": None,
            "acceptance_evidence_catalog_seal_hash": None,
            "acceptance_evidence_cutoff_at_utc": None,
            "operating_observation_snapshot_ref": None,
            "operating_observation_snapshot_hash": None,
            "adjustment_observation_snapshot_ref": None,
            "adjustment_observation_snapshot_hash": None,
            "scenario_control_snapshot_ref": None,
            "scenario_control_snapshot_hash": None,
            "valuation_control_snapshot_ref": None,
            "valuation_control_snapshot_hash": None,
            "valuation_comps_snapshot_ref": None,
            "valuation_comps_snapshot_hash": None,
            "compiler_version": "contract-fixture.v1",
            "template_version": "contract-fixture.v1",
            "fiscal_axis": FiscalAxis(
                historical_periods=(
                    FiscalPeriod(
                        period_id="FY2025",
                        fiscal_year=2025,
                        period_type="annual",
                        is_estimate=False,
                    ),
                ),
                forecast_periods=(
                    FiscalPeriod(
                        period_id="FY2026",
                        fiscal_year=2026,
                        period_type="annual",
                        is_estimate=True,
                    ),
                ),
            ),
            "acceptance_ancestry": DirectCanonicalBuildManifestAncestry(
                ancestry_kind="direct_canonical",
            ),
        }
    )
    return BuildManifest.model_validate(values, strict=True)


def _registration_token() -> WorkspaceProducerRegistrationToken:
    values: dict[str, object] = {
        "schema_version": "workspace-producer-registration-token.v1",
        "authority_scheme": "workspace_registry_v1",
        "workspace_scope_hash": _HASH_A,
        "registration_id": "bma603-contract-registration",
        "registration_version": 1,
        "registration_row_digest": _HASH_B,
    }
    values["token_digest"] = registration_token_digest(values)
    return WorkspaceProducerRegistrationToken.model_validate(values)


def _annual_data_lineage(manifest: BuildManifest) -> AnnualDataLineageBinding:
    values: dict[str, object] = {
        "schema_version": "annual-data-lineage-binding.v2",
        "annual_data_requirement_baseline_matrix_id": (
            manifest.annual_data_requirement_baseline_matrix_id
        ),
        "annual_data_requirement_baseline_matrix_version": (
            manifest.annual_data_requirement_baseline_matrix_version
        ),
        "annual_data_requirement_baseline_entries_digest": (
            manifest.annual_data_requirement_baseline_entries_digest
        ),
    }
    for prefix in _ANNUAL_DATA_PREFIXES:
        values[f"{prefix}_ref"] = getattr(manifest, f"{prefix}_ref")
        values[f"{prefix}_hash"] = getattr(manifest, f"{prefix}_hash")
    values["lineage_digest"] = annual_data_lineage_digest(values)
    return AnnualDataLineageBinding.model_validate(values)


def _closure(manifest: BuildManifest) -> AnnualManifestInputClosure:
    slot_registry_ref = _artifact_ref(
        "annual_manifest_slot_registry",
        global_scope=True,
    )
    producer_runtime_ref = _artifact_ref(
        "runtime_identity_snapshot",
        global_scope=True,
    )
    entries = tuple(
        AnnualManifestInputClosureEntry(
            input_slot=input_slot,
            binding_mode="required_successor_binding",
            selected_requirement_ids=(),
            artifact_refs=(),
            artifact_hashes=(),
            not_required_authority_bindings=(),
            successor_artifact_kind=f"{input_slot}_successor",
            required_before_gate="pre_render_acceptance",
        )
        for input_slot in ANNUAL_MANIFEST_INPUT_SLOTS
    )
    values: dict[str, object] = {
        "schema_version": "annual-manifest-input-closure.v1",
        "closure_id": "bma603-contract-closure",
        "closure_version": 1,
        "producer_contract_id": ANNUAL_MANIFEST_PRODUCER_CONTRACT_ID,
        "manifest_emission_id": "bma603-contract-emission",
        "emission_transaction_id": "bma603-contract-transaction",
        "model_state_ref": manifest.model_state_ref,
        "slot_registry_ref": slot_registry_ref,
        "slot_registry_hash": slot_registry_ref.content_sha256,
        "selected_requirement_set_ref": manifest.selected_requirement_set_ref,
        "selected_requirement_set_hash": manifest.selected_requirement_set_hash,
        "selected_annual_requirement_gap_projection_ref": (
            manifest.selected_annual_requirement_gap_projection_ref
        ),
        "selected_annual_requirement_gap_projection_hash": (
            manifest.selected_annual_requirement_gap_projection_hash
        ),
        "annual_data_lineage": _annual_data_lineage(manifest),
        "entries": entries,
        "unresolved_required_slots": (),
        "producer_registration_token": _registration_token(),
        "producer_runtime_identity_ref": producer_runtime_ref,
        "producer_runtime_identity_hash": producer_runtime_ref.content_sha256,
    }
    draft = AnnualManifestInputClosure.model_construct(
        **values,
        content_sha256=_HASH_A,
    )
    values["content_sha256"] = annual_manifest_input_closure_content_sha256(
        draft
    )
    return AnnualManifestInputClosure.model_validate(values, strict=True)


def build_bma603_annual_manifest_contract_world(
) -> Bma603AnnualManifestContractWorld:
    manifest = _manifest()
    return Bma603AnnualManifestContractWorld(
        manifest=manifest,
        closure=_closure(manifest),
    )
