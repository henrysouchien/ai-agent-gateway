# Contributing

## Development Setup

From the package root:

```bash
pip install -e ".[dev,anthropic,openai]"
```

If you only need one provider, install the matching extra instead.

## Running Tests

From the package root:

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
- use stable registry `model_key` values and install the matching provider extra
- use `create_gateway_app()` examples for custom assembly or approval scenarios
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

Release automation copies the complete package tree, including `docs/` and
`examples/`, into the standalone distribution. Keep package documentation
self-contained: do not link to files outside this tree.
