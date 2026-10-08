# agent_workflow_contracts

Dependency-neutral frozen pydantic wire models shared by the gateway, the
product agent runtimes, and any external consumer. No gateway imports —
that is the point of the split.

`AgentOperationSnapshot.execution_class` records the operation's work-kind.
Optional `role` and `effort` declarations participate in its semantic identity;
they do not resolve a model here. Both fields are omitted when `None`, preserving
historical snapshot dumps and sealed admitted-task digests.

| File | What it is |
|---|---|
| `models.py` | Frozen `WireModel` contracts including `CapabilityBind`, `AdmittedTask`, `TaskResult`, `DeliveryEnvelopeV1`/`V2`, `WorkflowView`, `PublishedOutput`, and `ContentHandle` |
| `ticker_contract.py`, `research_file_contract.py` | the two typed identifier contracts |
| `context_refs.py` | exact door-readback ref minter and declared-context renderer shared by pipeline stages and named children |
| `schema.py` | `public_json_schemas()` / `public_schema_bundle()` / `export_public_json_schemas()` |
| `generated/` | **build output** — JSON Schemas, `agent-workflow-contracts.d.ts`, the schema bundle, and delivery-envelope goldens. Regenerate from the checkout with `.venv/bin/python3 scripts/generate_agent_workflow_contracts.py`; do not hand-edit. |

Consumers and lifecycle: `../docs/api-reference.md` (section "Workflow Delivery Contracts")
The embedding application owns the workflow executor that admits and persists
these contracts.
