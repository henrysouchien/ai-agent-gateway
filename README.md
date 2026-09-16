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
| [`agent_gateway/heartbeat.py`](agent_gateway/heartbeat.py) | `HeartbeatLoop` | Repeated prebound autonomous work with quiet windows and backoff |
| [`agent_gateway/cli.py`](agent_gateway/cli.py) | `main()` (`agent init` / `agent run`) | Scaffold and run a package project |

The package does not infer a new model or credential inside autonomous
execution. The application resolves and passes the exact bound capability,
session, billing mode, and skill limits before calling `run_autonomous()`.

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
auth status <provider>` for all built-ins. Gateway token-store logout applies
to Anthropic and XAI; Codex credentials are managed by the Codex CLI, and
OpenAI uses `OPENAI_API_KEY`.

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

## Develop and verify

From the package directory, update a gateway dependency with:

```bash
pip-compile --generate-hashes --no-emit-options --allow-unsafe --upgrade-package <name>==<ver> --output-file requirements-dev.txt requirements-dev.in
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
