# canvas-kit-v1 type surface (vendored `node_modules`)

This tree is the complete declaration set the canvas typecheck resolves against. It is kept
on purpose: reviewed and accepted 2026-08-28 (walkthrough D15, option (a): keep and document).
Do not unvendor it, prune it, or swap it for an `npm install` without re-deciding D15.

## Why it exists

`agent_gateway/canvas_build_environment.py` typechecks every agent-authored canvas TSX with the
pinned `tsc` before bundling it. It builds that compile from `../tsconfig.typecheck.json`, pointing
`typeRoots` at `types/node_modules/@types` and `paths` at `@hank/canvas-kit`, `react` and
`recharts` in this tree. The check has to be **hermetic**: the same answer on a dev box, a pinned
gate and a production host, with no network and no `node_modules` of the checkout's own. So every
declaration the typecheck can reach is committed here and shipped in the wheel by the
`contracts/canvas-kit-v1/**/*.d.ts` and `**/*.json` package-data globs in
`packages/agent-gateway/pyproject.toml` (this README is not shipped).

What is here, and why each package is needed:

| Package | Why the typecheck needs it |
|---|---|
| `@hank/canvas-kit` | the kit's own declarations (`index.d.ts`, `fmt.d.ts`, `components/`), emitted by `tsc` from the kit source |
| `@types/react` | the `react` import every canvas makes; `paths` maps `react` here |
| `csstype` | imported by `@types/react` |
| `recharts` | `.d.ts` files only (the runtime is the host's `HankCanvasRuntime.Recharts` external) |
| `victory-vendor`, `@types/d3-scale`, `@types/d3-shape`, `@types/d3-path`, `@types/d3-time` | imported by recharts' declarations |
| `eventemitter3` | imported by recharts' declarations |
| `@types/lodash` | recharts' declarations import `lodash` types (`DebouncedFunc` in `types/chart/generateCategoricalChart.d.ts` and every `*Chart.d.ts`). `skipLibCheck` is `false`, so `tsc` must resolve them. It is by far the largest package here (702 of the 835 vendored files under `types/node_modules/` on 2026-09-23). |

`canvas_kit_manifest.v1.json` pins React, Recharts, TypeScript and esbuild; the declaration
packages above are whatever versions the Risk frontend resolved when the contract was last
generated (each package's own `package.json` is kept beside its `.d.ts` files and records it).

## Where it comes from

The whole `canvas-kit-v1/` directory is a byte-identical copy of the Risk repository's
`contracts/canvas_kit/v1/`, which that repository generates. The generator is
`risk_module/frontend/scripts/canvas-kit-contract/lib.mjs` (`emitTypes`): it runs `tsc` over
`frontend/packages/canvas-kit/tsconfig.contract.json` to emit `@hank/canvas-kit`, then copies
every `.d.ts` and `package.json` of the dependency list above out of the frontend's installed
`node_modules`, rewrites `canvas_kit_manifest.v1.json` and `tsconfig.typecheck.json`, and runs a
hermetic `tsc -p tsconfig.typecheck.json` over `fixtures/valid-canvas.tsx` before it writes.

## Regenerating

Do this when the kit's public surface, React, Recharts or the dependency list changes on the Risk
side — never by editing files in this tree.

1. In the Risk checkout, with the frontend's dependencies installed:

   ```bash
   cd risk_module/frontend
   npm run canvas:build     # rewrites contracts/canvas_kit/v1, src/generated, public/canvas-runtime
   npm run canvas:check     # regenerates into a temp dir and compares; must print "stable"
   ```

   Land that in Risk first.

2. In this checkout, replace this contract directory with Risk's, so deletions carry over too:

   ```bash
   rsync -a --delete --exclude /types/README.md \
     ../risk_module/contracts/canvas_kit/v1/ \
     packages/agent-gateway/agent_gateway/contracts/canvas-kit-v1/
   diff -rq ../risk_module/contracts/canvas_kit/v1 \
     packages/agent-gateway/agent_gateway/contracts/canvas-kit-v1
   # prints exactly one line: Only in …/canvas-kit-v1/types: README.md
   ```

3. Prove the copy still works here:

   ```bash
   .venv/bin/python3 -m pytest packages/agent-gateway/tests/test_canvas_kit_contract.py \
     packages/agent-gateway/tests/test_canvas_packaging.py \
     packages/agent-gateway/tests/test_canvas_build_environment.py
   ```

   and update the file count in the table above if it moved.

Nothing checks the two copies against each other automatically (the sync check was removed
in `4594f55375`); step 2's `diff` is the check. As of 2026-09-23 the two trees are identical
apart from this README.
