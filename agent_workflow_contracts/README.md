# agent_workflow_contracts

Dependency-neutral frozen pydantic wire models shared by the gateway, the
product agent runtimes, and any external consumer. No gateway imports —
that is the point of the split.

| File | What it is |
|---|---|
| `models.py` | Frozen `WireModel` contracts including `CapabilityBind`, `AdmittedTask`, `TaskResult`, `DeliveryEnvelopeV1`/`V2`, `WorkflowView`, `PublishedOutput`, and `ContentHandle` |
| `ticker_contract.py`, `research_file_contract.py` | the two typed identifier contracts |
| `schema.py` | `public_json_schemas()` / `public_schema_bundle()` / `export_public_json_schemas()` |
| `generated/` | **build output** — JSON Schemas, `agent-workflow-contracts.d.ts`, the schema bundle, and delivery-envelope goldens. Regenerate with `export_public_json_schemas()`; do not hand-edit. |

Consumers and lifecycle: `../docs/api-reference.md` (section "Workflow Delivery Contracts")
The embedding application owns the workflow executor that admits and persists
these contracts.
