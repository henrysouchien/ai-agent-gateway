# Contributing

## Development Setup

From `packages/agent-gateway/`:

```bash
pip install -e ".[dev,anthropic,openai]"
```

If you only need one provider, install the matching extra instead.

## Running Tests

From `packages/agent-gateway/`:

```bash
pytest tests
```

If you touch examples, it is also worth checking that the example entry points still parse:

```bash
python3 -m py_compile examples/*/agent.py
```

## Working On Docs And Examples

When you update docs in this package:

- keep code blocks copy-paste runnable
- prefer `curl` plus standard library Python over extra CLI dependencies such as `jq`
- say clearly when `create_agent()` is Anthropic-only
- use `create_gateway_app()` examples for multi-provider or custom approval scenarios
- note Docker preference and subprocess fallback anywhere code execution is shown

When you update examples:

- keep each example self-contained inside its directory
- include `README.md`
- include `agent.py`
- include `.env.example` when the example expects provider credentials

## Public API Changes

If you change the public API:

- update docstrings on exported symbols
- update [`README.md`](./README.md)
- update [`docs/api-reference.md`](./docs/api-reference.md)
- update any affected example directories

## MCP And Runtime Changes

If you change MCP, approval, or SSE behavior:

- update [`docs/http-api.md`](./docs/http-api.md)
- update [`docs/architecture.md`](./docs/architecture.md)
- verify the example clients still match the current approval flow

## Publishing And Sync

The package lives in this monorepo under `packages/agent-gateway/`.

The standalone distribution repo is synced with:

```bash
../../scripts/sync_agent_gateway.sh
```

That script copies the full package directory, including `docs/` and `examples/`.
