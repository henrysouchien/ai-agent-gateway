# ai-agent-gateway

Deploy an AI agent as an HTTP/SSE service without rebuilding session, tool,
approval, and execution infrastructure around the model loop. Requires Python
3.10 or newer.

## Ownership

This package owns the generic FastAPI chat surface, sessions and JWTs, SSE event
stream, provider/run loop, tool dispatch, approvals, code execution, skills,
sub-agents, autonomous execution, and heartbeat support.

It does not own an embedding product's profiles, prompts, channel policy,
domain tools, research data, or business schemas. Applications supply those
through `create_gateway_app()` or the higher-level `create_agent()` inputs.

The wheel imports without an application checkout. Product integrations bind
policy and identity callbacks plus optional batch/operator-schedule backends in
`GatewayServerConfig`; schema-backed artifact tools and routes belong to the
embedding application. `server_policy` is process-scoped: configure it before
building runtimes and use one product policy per gateway process.

## Main entrypoints

| Source | Primary symbol | Use it for |
| --- | --- | --- |
| [`agent_gateway/easy.py`](agent_gateway/easy.py) | `create_agent()` | A small server from a prompt, tools, skills, and provider configuration |
| [`agent_gateway/server.py`](agent_gateway/server.py) | `create_gateway_app()` | Product-owned runtime factories, auth, policy, and lifecycle integration |
| [`agent_gateway/gateway_server.py`](agent_gateway/gateway_server.py) | `run_gateway_server()` / `main()` | Privileged single-worker TCP launcher, claim-signing fd adoption, and startup-failure logging |
| [`agent_gateway/autonomous.py`](agent_gateway/autonomous.py) | `run_autonomous()` / `run_autonomous_sync()` | A prebound headless one-shot with no HTTP server |
| [`agent_gateway/admission.py`](agent_gateway/admission.py) | `admit()` | Shared spawn, continuation, message and handle authority |
| [`agent_gateway/heartbeat.py`](agent_gateway/heartbeat.py) | `HeartbeatLoop` | Repeated prebound autonomous work with quiet windows and backoff |
| [`agent_gateway/cli.py`](agent_gateway/cli.py) | `main()` (`agent init` / `agent run`) | Scaffold and run a package project |
| [`agent_gateway/gateway_address.py`](agent_gateway/gateway_address.py) | `gateway_base_url()` | The origin of the gateway a process talks to: the one reader of `GATEWAY_BASE_URL` and the one place its default is spelled |

`run_gateway_server()` is the serving entry on both local supported launches, not
only the privileged host one: the immutable runtime's serving child calls it
directly, so claim-signing adoption, `uvicorn.Config`/`Server` construction and
the launch-failure journal have one owner.

The package does not infer a new model or credential inside autonomous
execution. The application resolves and passes the exact bound capability,
session, billing mode, and skill limits before calling `run_autonomous()`.

Model selection and model request facts come from `model_authority`
(`hank-model-authority`; in the monorepo `packages/model-authority/`), which this
package depends on and carries no copy of: the registry, the selection policy,
each entry's per-family `compat` block and the rate tables.
`CapabilityExecutionResolver.select_bind()` in `capability_execution.py` selects
and authorizes a secret-free bind through that owner, without materializing
credentials or constructing an adapter. `resolve()` selects and materializes
once; `materialize_bind()` executes an already selected bind without reselection.
Both selection methods accept already-lowered `role`/exact `effort` and the
admission owner's frozen `selection_snapshot`. They use the shared
model-authority resolver; this adapter does not capture principal preferences
or own a map registry.
Bound executions retain their resolver for mid-run credential rotation.
`rebind_rotated_credential()` reauthorizes a new bind naming the selected
Anthropic source generation. Fresh spawns select and materialize through the
current scoped runner's execution resolver, so later descendants inherit the
rotated generation. Generation-bound executions do not rotate to material
without source-generation metadata.
[`agent_gateway/providers/`](agent_gateway/providers/README.md) builds each request
from the bound entry. Mechanism: `docs/reference/model-authority.md`.

`admit(SpawnRequest(...), parent, owners=AdmissionOwners(...))` returns the
selected context and exact `ResolvedAuthority`, without materializing credentials.
`admit(ContinueRequest(...), parent, owners=ContinuationOwners(...))` instead
retains the original `AdmittedTask`: no model reselection, semantic recompile, or
replacement tool grant. Its context preserves lineage and recorded spend;
`placement_lost` refreshes only backend placement/environment facts. Both results
carry the child's approval route from `child_approval_route()`: a child never
holds `DurableLocalApprovalRoute`; an in-process child records into its parent's
ledger through `InProcessDelegatedApprovalRoute`, a detached child adopts its
inherited channel as `ParentDelegatedApprovalRoute`, and a sessionless root's
children carry its `ControlPlaneApprovalRoute`. The existing task registry, transcript
reconstruction, attempt creation and credential materialization remain outside
this pure step. A valid older task without the context projection keeps its
durable authority; missing durable routes/definitions retain the resume owner's
existing handling.
`AdmissionOwners.frozen_bind` retains a plan-frozen model bind without
reselection; `materialize_bind()` is the single reauthorization owner. Fresh spawns select effort within the registered role; profile
overlays remain deferred. Declared operation children determine nested spawning.
Role narrowing reads the authenticated admitted tenant, not the presence or role
of an optional runtime session. The product policy still reads real session/profile
capability facts for its scoped exceptions; sessionless launch does not mint a
synthetic session or gain those exceptions.
Operation-private routes may extend the parent's visible surface, but cannot
bypass its withheld routes. Admission matches withheld canonical route IDs and
their registered policy names before freezing the executable grant.
The admitted wire retains `withheld_tools` across descendants and durable
continuation. Resume keeps its original executable grant while descendants
retain recorded exclusions plus the current parent's known exclusions.
`MessageRequest` and `HandleRequest` retain the caller's job context, using
spawn-passed handles plus current own/descendant outputs supplied by the durable
output owner. Message references are citations, never grants to the recipient.
The authenticated turn edge calls `root_context()` after approval, verified
research binding and withheld tools are ready; `current_parent_context()` reads
that captured scope without minting authority. A child session installs
`context_for_child(..., job_id=<physical task id>)` with `capture_job_context()`
and cannot regain root scope.
Sessionless control/schedule edges use `root_context_from_identity()` with their
authenticated identity, live roster and approval owner.
`TaskRegistry` recovery validates the canonical task and keeps its model bind as
execution authority. Duplicate registration/message bind receipts are optional
observations, not prerequisites or competing authority; canonical corruption and
task/result/message identity checks still apply.

Workflow-node continuation belongs to the workflow owner, not model-facing
`revive`. A trusted host first calls
`sub_agent.reserve_admitted_successor(runner, parked_task_id=...)` to reserve the
kernel's deterministic physical successor, then supplies its workflow admission
to `sub_agent.resume_admitted_successor(runner, parked_task_id=..., successor=...,
on_complete=...)`. The runner's dispatcher must contain the handler built by
`make_resume_handler`; this reuses its configured child-runtime wiring without
exposing supplied admissions through tool inputs. The admission retains the
parked logical task, authenticated tenant and frozen capability bind, names the
parked physical attempt in `resume_of_task_id`, uses the reserved successor ID as
both physical and admitted task ID, and carries the original grant unchanged
(including its grant ID and digest). Workflow attempt IDs and resume lifecycle events remain
workflow-owned. Completion releases the gateway's activity lease before invoking
the host callback, including an awaitable callback. Ordinary `revive` still
accepts only ordinary delegation identity and mints its existing ordinary
successor admission.

## Quick start

Install a provider extra and set its credential:

```bash
pip install "ai-agent-gateway[anthropic]"
export ANTHROPIC_API_KEY="your-anthropic-api-key"
export USER_DATA_DIR="$PWD/.agent-data"
mkdir -p "$USER_DATA_DIR/gateway"
chmod 700 "$USER_DATA_DIR" "$USER_DATA_DIR/gateway"
```

Create and run a project:

```bash
agent init my-agent
cd my-agent
agent run
```

Or create a FastAPI app directly:

```python
from agent_gateway import create_agent

app = create_agent(
    "You are a concise research assistant.",
    skills_dir="skills",
)
```

Serve the module with `uvicorn agent:app`. The default is the stable
`session.driver` entry in the configured model registry; applications can pass
another eligible stable `model_key`.

For OpenAI, install `ai-agent-gateway[openai]` and set `OPENAI_API_KEY`.
Managed provider login flows are available through `agent auth login
anthropic`, `agent auth login codex`, and `agent auth login xai`. Use `agent
auth status <provider>` for all built-ins. `agent auth login anthropic` is this
gateway's own Claude OAuth login: run it once per Claude account to enroll
interchangeable siblings, then `agent auth list anthropic` (masked) and `agent
auth remove anthropic <identity>` manage them. Gateway token-store logout
applies to Anthropic and XAI; Codex credentials are managed by the Codex CLI,
and OpenAI uses `OPENAI_API_KEY`.

The complete session-token, chat request, and SSE walkthrough is in the
[quickstart](docs/quickstart.md).

## Request shape

```text
HTTP client
  -> create_agent() or create_gateway_app()
  -> ChatRuntime
  -> AgentRunner.run() (provider stream and model/tool loop)
  -> ToolDispatcher.dispatch()
       |-- local Python handler
       |-- MCP stdio server
       |-- approval policy
       |-- code execution
       `-- run_agent sub-agent
  -> EventLog -> SSE response
```

Headless calls use the same runner, tools, skills, and provider abstractions,
but return `RunOutput` instead of exposing an HTTP/SSE session.

Code execution settles native and bridged nested calls through the cell ledger.
Their `dispatch.sources` remain in the outer result's `sources`; native computation
provenance is projected into `computations` with its `nested_call_id`, without
entry or byte caps. Computation JSON sidecars are no longer read or consumed.

Native calls parked for local approval capture their original `PreparedToolCall`
and dispatch operands privately on the route's ledger store, before publication.
Custody uses the executing job's admitted context (including in-process children),
and an Anthropic generation reference comes only from the dispatcher's frozen
capability bind. A captured root turn links that same park to its root admission.
Automatic decisions and detached delegated approval do not record a native
binding; a missing admitted job or tenant is observed with a warning and approval continues.
Public approval projections remain redacted and carry none of these private operands.

## Develop and verify

From the repository root, update a gateway dependency with the constrained
target (the dev lock is compiled `--constraint api/requirements.txt` so the two
locks cannot disagree; `docs/reference/python-environments-and-packaging.md`):

```bash
make gateway-deps-update PKG=<name>   # then: make gateway-deps-install
```

For coupled targets, use multiple space-separated `--upgrade-package <name>==<ver>`
flags. In the monorepo checkout, the shortcut is
`make gateway-deps-update PKG='<name>==<ver>'` (`PKG` accepts a space-separated list).

From the package root, run:

```bash
pytest tests
```

The suite and its `gateway_test_support/` fixtures are package-local; no embedding
application checkout or product configuration is required. Product integration
scenarios belong in the embedding application's tests, not this package suite.

Use the focused test for the boundary you change. Runnable examples live under
[`examples/`](examples/); the production-style assembly example is
[`examples/07-full-production/`](examples/07-full-production/), and the
headless fixture is [`examples/09-autonomous/`](examples/09-autonomous/).

## Documentation

- [Quickstart](docs/quickstart.md)
- [Architecture](docs/architecture.md)
- [HTTP and SSE API](docs/http-api.md)
- [Python API reference](docs/api-reference.md)
- [MCP server configuration](docs/mcp-server-catalog.md)
- [Framework comparison](docs/comparison.md)
- [Contributing](CONTRIBUTING.md)

## License

MIT
