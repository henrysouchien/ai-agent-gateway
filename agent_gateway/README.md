# agent_gateway — source map

Source map for the gateway runtime package. This file is a map only; the
library API is documented in `../docs/`.

| Subpackage | What it holds |
|---|---|
| `control_plane/` | the generic `/control/*` FastAPI routers (session, profiles, skills, schedules, runs, batches, approvals, events, health); product backends are supplied explicitly |
| `code_execution/_backends/` | `ExecutionBackend` placements of one session-sticky Python kernel: `KernelBackend` (`placement="local"`) and `DockerBackend` (`placement="docker"`) share the persistent runner/codec |
| `canvas_build/` | pinned Node toolchain (`.node-version`, `provision.sh`, `build.mjs`, `policy.mjs`) that compiles agent-authored canvas TSX |
| `contracts/canvas-kit-v1/` | the contract `canvas_build` compiles against: `canvas_kit_manifest.v1.json` (externals, pinned versions, size caps) + 7 conformance fixtures whose expected stage/code live in `fixtures/expectations.json`. The `@hank/canvas-kit` type surface is committed under `types/node_modules/` and shipped by the `contracts/canvas-kit-v1/**/*.d.ts` package-data glob in `pyproject.toml`. |
| `contracts/` (others) | `claim-contract-v1`, `completion-record-v1`, `control-run-v1`, `ui-blocks-v1`, `usage-reconciliation-v{1,2,3}` JSON Schemas and fixtures |
| `multi_user/` | the per-user usage ledger and its dead-letter spool |

The model authority — registry, selection policy, resolver, rate tables, effort
vocabulary and `CapabilityBind` — is not in this package. It is
`hank-model-authority` (`../../model-authority/`, import `model_authority`).

Provider adapters, credential stores, transport types and one-shot completion
helpers live in [`hank_model_transport`](../../model-transport/hank_model_transport/),
not in the gateway. Its adapters build `ModelInfo` from the model registry entry,
its `compat` block and rate tables; a model ID with no registry entry is not callable.
The gateway owns `AuthConfig`, capability resolvers, and
`transport_runtime.GATEWAY_SEND_PROMPT_RUNTIME`, which gateway-hosted one-shot
callers pass explicitly as `runtime=` to the transport's `send_prompt()` or
`send_prompt_sync()`. Gateway host metadata lives in `gateway_info.py`; the
transport's `package_info.py` holds only transport identity and version.

Event credential projection: `secret_boundary.py` redacts authored inputs and
display text in both `tool_calls_planned` and call-start events, leaving opaque
call IDs and tool names unchanged so consumers can correlate the same call.

Library reference: `../docs/architecture.md`, `../docs/api-reference.md`, `../docs/http-api.md`
Operator view: the control-plane modules above and the generated OpenAPI
document are authoritative for generic `/control` routes. Hank's schema-backed artifact routes and stores live in [`api/agent/shared`](../../../api/agent/shared), outside this library.
Wire contracts: `../../agent-workflow-contracts/agent_workflow_contracts/README.md`
