# agent_gateway — source map

Source map for the gateway runtime package. This file is a map only; the
library API is documented in `../docs/`.

| Subpackage | What it holds |
|---|---|
| `providers/` | model adapters — `anthropic`, `codex`, `openai`, `xai`, `agent_sdk`, plus `*_helpers` and the `anthropic_oauth` / `xai_oauth` token stores |
| `control_plane/` | the generic `/control/*` FastAPI routers (session, profiles, skills, schedules, runs, batches, approvals, events, health); product backends are supplied explicitly |
| `code_execution/_backends/` | the two `ExecutionBackend` implementations, `DockerBackend` and `SubprocessBackend` |
| `canvas_build/` | pinned Node toolchain (`.node-version`, `provision.sh`, `build.mjs`, `policy.mjs`) that compiles agent-authored canvas TSX |
| `contracts/canvas-kit-v1/` | the contract `canvas_build` compiles against: `canvas_kit_manifest.v1.json` (externals, pinned versions, size caps) + 7 conformance fixtures whose expected stage/code live in `fixtures/expectations.json`. The `@hank/canvas-kit` type surface is committed under `types/node_modules/` and shipped by the `contracts/canvas-kit-v1/**/*.d.ts` package-data glob in `pyproject.toml`. |
| `contracts/` (others) | `claim-contract-v1`, `commercial-*`, `control-run-v1`, `ui-blocks-v1`, `usage-reconciliation-v{1,2,3}` JSON Schemas and fixtures |
| `multi_user/` | the per-user usage ledger and its dead-letter spool |
| `model_authority/` | `product-model-registry.yaml` — the typed `model_key` → model resolution |
| `rates/` | provider price tables |

Library reference: `../docs/architecture.md`, `../docs/api-reference.md`, `../docs/http-api.md`
Operator view: the control-plane modules above and the generated OpenAPI
document are authoritative for generic `/control` routes. Hank's schema-backed artifact routes and stores live in `api/agent/shared`, outside this library.
Wire contracts: `../agent_workflow_contracts/README.md`
